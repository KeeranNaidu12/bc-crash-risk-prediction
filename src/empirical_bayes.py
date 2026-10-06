"""
empirical_bayes.py
Data loading and the Empirical Bayes model, shared by 03_train_models.py (forecast)
and 04_evaluate_model.py (diagnostics).

Empirical Bayes (the Highway Safety Manual method) blends two estimates:
  typical   crashes expected at an intersection like this one (negative binomial
            regression on its features, a "safety performance function")
  history   this intersection's own average crashes per year
and trusts the history more when the intersection is expected to have many crashes.

ICBC only lists intersections that have had a crash, so every intersection with history
has at least one crash in it. The regression is therefore ZERO-TRUNCATED: it is fitted
on each intersection's total crashes so far, allowing for the fact that totals of zero
are never seen, and so estimates typical crashes for ALL intersections of a kind. A
regression fitted only on intersections with crashes would already count the crash that
put them in the data, and Empirical Bayes would count it again, over-predicting quiet
intersections.

Crash levels across the city change from year to year (they rose after 2021), so the
forecast is scaled to carry the most recent year's level forward: summed over all the
intersections predicted, it equals their crashes in the most recent year (a calibration
factor, as in the Highway Safety Manual).
"""
import numpy as np
import pandas as pd
import patsy
from scipy.special import gammaln
from scipy.stats import nbinom
from statsmodels.base.model import GenericLikelihoodModel

from db import get_engine

TEST_YEAR = 2025

# Few intersections have more than 4 streets, so 4+ are grouped together (and the one
# single-street location is grouped with 2). Street class (the most major street meeting
# there) stands in for traffic volume; local streets are the reference level.
SPF_FEATURES = ("is_signalized + is_interchange + C(n_streets.clip(2, 4))"
                " + C(street_class, Treatment('local'))")
SPF_MAXITER = 1000


def load_raw() -> pd.DataFrame:
    """Crash counts per intersection per year, with the intersection's features."""
    raw = pd.read_sql("""
        SELECT y.location, y.year, y.crashes,
               f.latitude, f.longitude, f.n_streets, f.is_interchange, f.is_signalized,
               f.street_class
        FROM intersection_year y
        JOIN intersection_features f USING (location)
    """, get_engine())
    raw[["is_interchange", "is_signalized"]] = raw[["is_interchange", "is_signalized"]].astype(int)
    return raw


def add_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training and testing, with history from PREVIOUS years only."""
    df = raw.sort_values(["location", "year"])
    history = df.groupby("location")
    df["prior_crashes"] = history["crashes"].shift(1)
    df["prior_avg"] = history["crashes"].transform(lambda s: s.shift(1).expanding().mean())
    df["prior_years"] = history.cumcount()
    df = df.dropna(subset=["prior_crashes"])  # 2021 has no previous year

    # ICBC only lists intersections that had a crash, so an intersection whose first crash
    # is in year Y would only be in the data because of year Y. Keep a row only if the
    # intersection had a crash in an EARLIER year, i.e. it was known when predicting.
    df["known_before"] = history["crashes"].transform(lambda s: s.shift(1).cumsum()) > 0
    return df[df["known_before"]]


def forecast_rows(raw):
    """One row per intersection with all years as history, for forecasting the next year."""
    return (raw.groupby("location")
               .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                    n_streets=("n_streets", "first"), is_interchange=("is_interchange", "first"),
                    is_signalized=("is_signalized", "first"),
                    street_class=("street_class", "first"),
                    prior_avg=("crashes", "mean"), prior_years=("crashes", "size"))
               .reset_index())


def level_factor(raw, locations, year):
    """Most recent year's crashes over the average year's, using only years before `year`.

    The history average assumes crash levels are flat; this factor carries the latest
    level forward instead. It is computed over the intersections being predicted.
    """
    totals = raw[raw["location"].isin(locations) & (raw["year"] < year)].groupby("year")["crashes"].sum()
    return totals.iloc[-1] / totals.mean()


def year_variation(raw, locations, year):
    """How much an intersection's crash rate drifts from year to year, beyond chance.

    Each year's counts are divided by that year's citywide level, then phi solves
    variance = mean + phi * mean^2 across each intersection's years (method of moments),
    using only years before `year`. Empirical Bayes treats each intersection's rate as
    fixed, so without phi the 90% ranges at busy intersections are too narrow.
    """
    history = raw[raw["location"].isin(locations) & (raw["year"] < year)]
    totals = history.groupby("year")["crashes"].sum()
    relative = history["crashes"] / history["year"].map(totals / totals.mean())
    by_location = relative.groupby(history["location"])
    mean, var = by_location.mean(), by_location.var()
    return max((var - mean).sum() / (mean ** 2).sum(), 0.0)


class ZeroTruncatedNB(GenericLikelihoodModel):
    """Negative binomial regression of total crashes over `exposure` years, for data in
    which only totals of 1 or more are seen. The last parameter is log k (overdispersion),
    so that k stays positive."""

    def __init__(self, endog, exog, exposure, **kwargs):
        super().__init__(endog, exog, extra_params_names=["log_k"], **kwargs)
        self.exposure = np.asarray(exposure, dtype=float)

    def loglikeobs(self, params):
        beta, k = params[:-1], np.exp(params[-1])
        mean, size, y = self.exposure * np.exp(self.exog @ beta), 1 / k, self.endog
        log_p_zero = size * np.log(size / (size + mean))
        log_nb = (gammaln(y + size) - gammaln(size) - gammaln(y + 1)
                  + log_p_zero + y * np.log(mean / (size + mean)))
        return log_nb - np.log(-np.expm1(log_p_zero))  # divide by P(total >= 1)


def fit_spf(rows, method="bfgs"):
    """Typical crashes per year for each kind of intersection (safety performance function).

    Fitted on each row's HISTORY (prior_avg x prior_years crashes over prior_years years),
    never on the row's own crashes, so it can be fitted on the very rows being predicted.
    """
    exog = patsy.dmatrix(SPF_FEATURES, rows, return_type="dataframe")
    total = rows["prior_avg"] * rows["prior_years"]
    model = ZeroTruncatedNB(total, exog, exposure=rows["prior_years"])
    model.design_info = exog.design_info
    start = np.r_[np.log(total.sum() / rows["prior_years"].sum()), np.zeros(exog.shape[1])]
    return model.fit(start_params=start, method=method, maxiter=SPF_MAXITER, disp=0)


def coefficients(spf):
    """The regression's parameters, named (statsmodels returns them as a bare array here)."""
    return pd.Series(spf.params, index=spf.model.exog_names)


def overdispersion(spf):
    """k: how much intersections of the same kind vary beyond chance."""
    return float(np.exp(coefficients(spf)["log_k"]))


def typical_rate(spf, rows):
    """Typical crashes per year (at the average year's level) for intersections like these."""
    exog = patsy.build_design_matrices([spf.model.design_info], rows, return_type="dataframe")[0]
    return pd.Series(np.exp(exog.to_numpy() @ coefficients(spf).drop("log_k").to_numpy()), index=rows.index)


def empirical_bayes(spf, rows, k=None, level=1.0, phi=0.0):
    """Blend typical crashes with each intersection's own history.

    The regression gives a Gamma prior for each intersection's true crash rate
    (mean = typical, variance = k * typical^2). Updating it with the observed crashes
    gives a Gamma posterior; its mean is the Empirical Bayes estimate, and the matching
    negative binomial gives a 90% range for next year's count.

    `level` is next year's crash level over the history average (see level_factor). The
    posterior rates are scaled so that their total is `level` x the history total, i.e. the
    most recent year's crashes. (Without the scaling the total would fall short, because
    typical includes intersections that never crash.) Typical gets the same scaling, so
    typical and predicted are both for next year, and so is excess (predicted - typical). The range
    also allows for the rate drifting from year to year (phi, see year_variation): the
    count's distribution is a negative binomial with the matching mean and variance.

    k is the regression's overdispersion unless given (04_evaluate_model.py varies it
    to check how much shrinkage is best).
    """
    if k is None:
        k = overdispersion(spf)
    typical = typical_rate(spf, rows)
    shape = 1 / k + rows["prior_avg"] * rows["prior_years"]
    rate = 1 / (k * typical) + rows["prior_years"]
    level = level * rows["prior_avg"].sum() / (shape / rate).sum()  # calibration factor

    predicted = shape / rate * level
    rate_variance = shape / rate ** 2 * level ** 2  # uncertainty about next year's rate
    variance = predicted + rate_variance + phi * (rate_variance + predicted ** 2)
    nb_shape = predicted ** 2 / (variance - predicted)
    nb_p = nb_shape / (nb_shape + predicted)
    return pd.DataFrame({
        "typical": typical * level,
        "predicted": predicted,
        "low_90": nbinom.ppf(0.05, nb_shape, nb_p),
        "high_90": nbinom.ppf(0.95, nb_shape, nb_p),
        "weight": 1 / (1 + k * typical * rows["prior_years"]),  # weight on typical
        "nb_shape": nb_shape,  # next year's count is negative binomial(nb_shape, nb_p)
        "nb_p": nb_p,
    }, index=rows.index)
