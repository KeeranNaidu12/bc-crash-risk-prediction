"""
03_train_models.py
Predict next year's crashes per intersection with Empirical Bayes, check it against
simple baselines on past years, and save a 2026 forecast for Tableau.

Empirical Bayes (the Highway Safety Manual method) blends two estimates:
  typical   crashes expected at an intersection like this one (negative binomial
            regression on its features, a "safety performance function")
  history   this intersection's own average crashes per year
and trusts the history more when the intersection is expected to have many crashes.

Intersections are ranked by EXCESS crashes (Empirical Bayes estimate minus typical):
how many more crashes a year an intersection has than is normal for its type. This
finds intersections that underperform, rather than just the busiest ones.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import nbinom
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from db import get_engine

ROOT = Path(__file__).resolve().parents[1]
TEST_YEAR = 2025
FORECAST_YEAR = TEST_YEAR + 1

# Few intersections have more than 4 streets, so 4+ are grouped together (and the one
# single-street location is grouped with 2)
SPF_FORMULA = "crashes ~ is_signalized + is_interchange + C(n_streets.clip(2, 4))"


# 1. Load crash counts and intersection features from PostgreSQL
raw = pd.read_sql("""
    SELECT y.location, y.year, y.crashes,
           f.latitude, f.longitude, f.n_streets, f.is_interchange, f.is_signalized
    FROM intersection_year y
    JOIN intersection_features f USING (location)
""", get_engine())
raw[["is_interchange", "is_signalized"]] = raw[["is_interchange", "is_signalized"]].astype(int)

# 2. History: only use PREVIOUS years, so the model can't see the answer
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
df = df[df["known_before"]]


# 3. Empirical Bayes
def fit_spf(train):
    """Negative binomial regression: typical crashes per year for each kind of intersection."""
    return smf.negativebinomial(SPF_FORMULA, data=train).fit(disp=0)


def empirical_bayes(spf, rows):
    """Blend typical crashes with each intersection's own history.

    The regression gives a Gamma prior for each intersection's true crash rate
    (mean = typical, variance = k * typical^2). Updating it with the observed crashes
    gives a Gamma posterior; its mean is the Empirical Bayes estimate, and the matching
    negative binomial gives a 90% range for next year's count.
    """
    k = spf.params["alpha"]  # overdispersion: how much intersections vary beyond chance
    typical = spf.predict(rows)
    shape = 1 / k + rows["prior_avg"] * rows["prior_years"]
    rate = 1 / (k * typical) + rows["prior_years"]
    p = rate / (rate + 1)
    return pd.DataFrame({
        "typical": typical,
        "predicted": shape / rate,
        "low_90": nbinom.ppf(0.05, shape, p),
        "high_90": nbinom.ppf(0.95, shape, p),
        "weight": 1 / (1 + k * typical * rows["prior_years"]),  # weight on typical
    }, index=rows.index)


def top_k_crash_share(actual, predicted, k=100):
    """Share of the year's crashes that happened at the k intersections predicted riskiest."""
    return actual[predicted.nlargest(k).index].sum() / actual.sum()


# 4. Check on past years, one year at a time: fit on every year before Y, predict Y.
#    2023 and 2024 are validation years (use them to compare methods and settings);
#    2025 is the final test and should only be looked at once changes are settled.
print(f"{'':16} {'':22} {'MAE':>5} {'Poisson dev':>11} {'top-100 share':>13} {'in 90% range':>12}")
for year in range(df["year"].min() + 1, TEST_YEAR + 1):
    train = df[df["year"] < year]
    test = df[df["year"] == year].copy()
    test = test.join(empirical_bayes(fit_spf(train), test))
    covered = test["crashes"].between(test["low_90"], test["high_90"]).mean()

    label = "TEST" if year == TEST_YEAR else "validation"
    for name, pred in [("Same as last year", test["prior_crashes"]),
                       ("Average of prior years", test["prior_avg"]),
                       ("Empirical Bayes", test["predicted"])]:
        mae = mean_absolute_error(test["crashes"], pred)
        deviance = mean_poisson_deviance(test["crashes"], pred.clip(lower=0.01))
        share = top_k_crash_share(test["crashes"], pred)
        coverage = f"{covered:12.1%}" if name == "Empirical Bayes" else ""
        print(f"{year} {label:10} {name:22} {mae:5.2f} {deviance:11.2f} {share:13.1%} {coverage}")

# 5. Forecast next year from all years of history
spf = fit_spf(df)
latest = (raw.groupby("location")
             .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                  n_streets=("n_streets", "first"), is_interchange=("is_interchange", "first"),
                  is_signalized=("is_signalized", "first"),
                  prior_avg=("crashes", "mean"), prior_years=("crashes", "size"))
             .reset_index())
forecast = latest.join(empirical_bayes(spf, latest))
forecast["excess"] = forecast["predicted"] - forecast["typical"]

# 6. Explain the model: how each feature multiplies typical crashes per year
names = {"is_signalized": "signalized", "is_interchange": "interchange",
         "C(n_streets.clip(2, 4))[T.3]": "3 streets", "C(n_streets.clip(2, 4))[T.4]": "4+ streets"}
print(f"\nTypical crashes per year at an unsignalized 2-street intersection: {np.exp(spf.params['Intercept']):.1f}")
for term, name in names.items():
    print(f"  {name:12} x{np.exp(spf.params[term]):.2f}")
print(f"Median weight on typical (vs the intersection's own history): {forecast['weight'].median():.2f}")

# 7. Save the forecast for Tableau, ranked by excess crashes
output = forecast[["location", "latitude", "longitude", "is_signalized", "prior_avg",
                   "typical", "predicted", "low_90", "high_90", "excess"]].copy()
output = output.rename(columns={"prior_avg": "avg_crashes_2021_2025", "typical": "typical_for_type",
                                f"predicted": f"predicted_{FORECAST_YEAR}"})
output = output.round(1)
output["excess_rank"] = output["excess"].rank(ascending=False, method="first").astype(int)
output["predicted_rank"] = output[f"predicted_{FORECAST_YEAR}"].rank(ascending=False, method="first").astype(int)
output = output.sort_values("excess_rank")
output.to_csv(ROOT / "dashboard" / "predictions.csv", index=False)
print(f"\nTop 10 intersections by excess crashes ({FORECAST_YEAR} forecast):")
print(output.head(10)[["location", f"predicted_{FORECAST_YEAR}", "low_90", "high_90",
                       "typical_for_type", "excess"]].to_string(index=False))
print("Saved dashboard/predictions.csv")
