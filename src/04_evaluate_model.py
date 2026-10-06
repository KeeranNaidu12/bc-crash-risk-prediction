"""
04_evaluate_model.py
Diagnostics for the Empirical Bayes model in empirical_bayes.py. Prints three checks:

  1. Convergence   did each negative binomial fit reach a true maximum?
  2. Shrinkage     is the blend of typical and history weighted about right?
  3. Calibration   do predicted counts and 90% ranges match what happened?

Every check uses the same year-by-year setup as 03_train_models.py: fit on every year
before Y, predict Y, for Y = 2023, 2024 (validation) and 2025 (test); the regression
is fitted on the history of the intersections being predicted. The plain average
it is compared with gets the same level factor as Empirical Bayes ("average x level").

Usage (from the project root, after 03_train_models.py):
    python src/04_evaluate_model.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.metrics import mean_poisson_deviance

from empirical_bayes import (SPF_MAXITER, TEST_YEAR, add_history, coefficients,
                             empirical_bayes, fit_spf, forecast_rows, level_factor, load_raw,
                             overdispersion, year_variation)

rng = np.random.default_rng(42)
raw = load_raw()
df = add_history(raw)
years = range(df["year"].min() + 1, TEST_YEAR + 1)


def deviance(actual, predicted):
    return mean_poisson_deviance(actual, predicted.clip(lower=0.01))


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# Fit once per fold (and once on all years, as the forecast does); every check reuses these
folds = {}
for year in years:
    test = df[df["year"] == year].copy()
    spf = fit_spf(test)  # on the history before `year` only
    test["level"] = level_factor(raw, test["location"], year)
    test["phi"] = year_variation(raw, test["location"], year)
    test["avg_level"] = test["prior_avg"] * test["level"]
    eb = empirical_bayes(spf, test, level=test["level"].iloc[0], phi=test["phi"].iloc[0])
    folds[year] = (test, spf, test.join(eb))
latest = forecast_rows(raw)
fits = {f"history < {y}": (rows, spf) for y, (rows, spf, _) in folds.items()}
fits["all years (forecast)"] = (latest, fit_spf(latest))


# 1. Convergence -------------------------------------------------------------------------
section("1. OPTIMIZATION CONVERGENCE (negative binomial regression)")
print("A fit has converged to a true maximum if the optimizer says so, the gradient is ~0,\n"
      "the Hessian is negative definite (every eigenvalue of -H > 0), and a different\n"
      "optimizer lands on the same answer. The zero-truncated regression is fitted with BFGS;\n"
      "L-BFGS is the cross-check, and Newton is also tried as the usual alternative.\n")
print(f"{'':22} {'conv':>5} {'calls':>5} {'max|grad|':>10} {'min eig(-H)':>12} "
      f"{'cond(-H)':>9} {'L-BFGS dLL':>11} {'max dparam':>11} {'Newton':>9}")
for name, (rows, spf) in fits.items():
    retvals = spf.mle_retvals
    grad = np.abs(spf.model.score(spf.params)).max()
    eig = np.linalg.eigvalsh(-spf.model.hessian(spf.params))
    alt = fit_spf(rows, method="lbfgs")
    with warnings.catch_warnings():  # a failed Newton fit warns; the table reports it
        warnings.simplefilter("ignore")
        newton = fit_spf(rows, method="newton")
    newton_ok = newton.mle_retvals["converged"] and np.isfinite(newton.llf)
    newton_result = f"{newton.llf - spf.llf:9.1e}" if newton_ok else f"{'failed':>9}"
    print(f"{name:22} {str(retvals['converged']):>5} {retvals['fcalls']:5d} {grad:10.1e} "
          f"{eig.min():12.2e} {eig.max() / eig.min():9.1e} {alt.llf - spf.llf:11.1e} "
          f"{np.abs(alt.params - spf.params).max():11.1e} {newton_result}")
print(f"(calls = log-likelihood evaluations, iteration limit {SPF_MAXITER}; dLL = other optimizer's log-likelihood minus BFGS's,\n"
      " so dLL > 0 would mean BFGS stopped short of the maximum)")

spf = fits["all years (forecast)"][1]
names = coefficients(spf).index
ci = pd.DataFrame(spf.conf_int(), index=names)
print(f"\nOverdispersion k (all years): {overdispersion(spf):.3f}  "
      f"95% CI {np.exp(ci.loc['log_k', 0]):.3f}-{np.exp(ci.loc['log_k', 1]):.3f}")
print("Coefficients (all years), as multipliers on typical crashes with 95% CI:")
coef = pd.DataFrame({"mult": np.exp(coefficients(spf)), "low": np.exp(ci[0]),
                     "high": np.exp(ci[1]), "p": pd.Series(spf.pvalues, index=names)}).drop("log_k")
coef.index = coef.index.str.replace(r"C\((\w+).*\)\[T\.(.+)\]", r"\1=\2", regex=True)
print(coef.round(3).to_string())


# 2. Shrinkage ---------------------------------------------------------------------------
section("2. SHRINKAGE BEHAVIOUR")
print("Weight on typical = 1 / (1 + k * typical * years of history). 0 = own history only,\n"
      "1 = typical only. Below: median weight in each test year, by street class.\n")
weights = pd.concat({y: t.groupby("street_class")["weight"].median() for y, (_, _, t) in folds.items()},
                    axis=1)
weights.loc["all"] = [t["weight"].median() for _, _, t in folds.values()]
print(weights.round(3).to_string())

print("\nIs the amount of shrinkage right? k is scaled by c, and Poisson deviance recorded.\n"
      "c < 1 trusts typical more, c > 1 trusts history more. If c = 1 (the fitted k) is at\n"
      "or near the lowest deviance, the regression's overdispersion gives the right blend.\n")
scales = [0.25, 0.5, 0.75, 1, 1.5, 2, 4]
rows = {}
for year, (_, spf, test) in folds.items():
    k = overdispersion(spf)
    row = {"typical only": deviance(test["crashes"], test["typical"])}
    row.update({f"c={c}": deviance(test["crashes"], empirical_bayes(spf, test, k=k * c,
                                                                       level=test["level"].iloc[0])["predicted"])
                for c in scales})
    row["history only"] = deviance(test["crashes"], test["avg_level"])
    rows[year] = row
shrink = pd.DataFrame(rows).T
print(shrink.round(3).to_string())
print("Lowest deviance per year:", ", ".join(f"{y}: {shrink.loc[y].idxmin()}" for y in shrink.index))

print("\nRegression to the mean: mean crashes at the 10% of intersections with the highest and\n"
      "lowest average so far. A plain average overshoots both ways; EB should pull back.\n")
print(f"{'':6} {'group':12} {'n':>5} {'actual':>7} {'avg x level':>12} {'EB':>7}")
for year, (_, _, test) in folds.items():
    top = test["prior_avg"] >= test["prior_avg"].quantile(0.9)
    low = test["prior_avg"] <= test["prior_avg"].quantile(0.1)
    for group, mask in [("top 10%", top), ("bottom 10%", low)]:
        t = test[mask]
        print(f"{year:6} {group:12} {len(t):5d} {t['crashes'].mean():7.2f} "
              f"{t['avg_level'].mean():12.2f} {t['predicted'].mean():7.2f}")

print("\nWhere shrinkage helps: deviance of EB vs average x level, by weight on typical\n"
      "(all three test years pooled). EB should gain most where it shrinks most.\n")
pooled = pd.concat([t for _, _, t in folds.values()])
pooled["weight_bin"] = pd.cut(pooled["weight"], [0, 0.05, 0.1, 0.2, 0.4, 1])
print(f"{'weight on typical':18} {'n':>6} {'EB dev':>7} {'avg dev':>8} {'EB better by':>13}")
for bin_, t in pooled.groupby("weight_bin", observed=True):
    eb, avg = deviance(t["crashes"], t["predicted"]), deviance(t["crashes"], t["avg_level"])
    print(f"{str(bin_):18} {len(t):6d} {eb:7.3f} {avg:8.3f} {(avg - eb) / avg:13.1%}")


# 3. Calibration -------------------------------------------------------------------------
section("3. PREDICTIVE CALIBRATION")
print("Overall bias: total predicted vs total actual crashes in each test year.\n")
print(f"{'':6} {'level':>6} {'actual':>8} {'EB':>8} {'EB/actual':>10} {'avg x level':>12} {'/actual':>8}")
for year, (_, _, test) in folds.items():
    actual, eb, avg = test["crashes"].sum(), test["predicted"].sum(), test["avg_level"].sum()
    print(f"{year:6} {test['level'].iloc[0]:6.3f} {actual:8.0f} {eb:8.0f} {eb / actual:10.3f} "
          f"{avg:12.0f} {avg / actual:8.3f}")

print("\nCalibration by predicted level (all test years pooled, deciles of EB prediction).\n"
      "Calibrated if mean actual matches mean predicted in every decile.\n")
pooled["decile"] = pd.qcut(pooled["predicted"], 10, labels=False) + 1
calib = pooled.groupby("decile").agg(n=("crashes", "size"), predicted=("predicted", "mean"),
                                     actual=("crashes", "mean"))
calib["actual/predicted"] = calib["actual"] / calib["predicted"]
print(calib.round(3).to_string())

print("\nInterval coverage: share of actual counts inside the central X% range of the\n"
      "predicted negative binomial. Should be close to X, but whole-number counts push it\n"
      "above X where crashes are rare (with a mean of 0.5, the range 0-2 already holds ~98%),\n"
      "so the PIT tails below are the fairer check by street class.\n")
levels = [0.5, 0.8, 0.9, 0.95]


def coverage(t, level):
    low = nbinom.ppf((1 - level) / 2, t["nb_shape"], t["nb_p"])
    high = nbinom.ppf(1 - (1 - level) / 2, t["nb_shape"], t["nb_p"])
    return t["crashes"].between(low, high).mean()


cov = pd.DataFrame({f"{l:.0%}": {**{y: coverage(t, l) for y, (_, _, t) in folds.items()},
                                  **{c: coverage(t, l) for c, t in pooled.groupby("street_class")}}
                    for l in levels})
print((cov * 100).round(1).to_string())

print("\nRandomized PIT (all test years pooled): where each actual count falls in its\n"
      "predicted distribution, from 0 to 1. Calibrated forecasts give ~10% in every bin;\n"
      "too many at both ends = ranges too narrow, too many in the middle = too wide.\n")
y, shape, p = pooled["crashes"], pooled["nb_shape"], pooled["nb_p"]
pit = nbinom.cdf(y - 1, shape, p) + rng.uniform(size=len(y)) * nbinom.pmf(y, shape, p)
counts, edges = np.histogram(pit, bins=10, range=(0, 1))
for count, left in zip(counts, edges):
    share = count / len(pit)
    print(f"  {left:.1f}-{left + 0.1:.1f}  {share:6.1%}  {'#' * round(share * 200)}")

print("\nPIT tails by street class: share of counts below the 5th or above the 95th percentile\n"
      "of their predicted distribution. Calibrated = 5% each side, 10% in all; more means\n"
      "ranges too narrow, fewer means too wide.\n")
pooled["pit"] = pit
tails = pooled.groupby("street_class")["pit"].agg(n="size", low=lambda s: (s < 0.05).mean(),
                                                   high=lambda s: (s > 0.95).mean())
tails["both"] = tails["low"] + tails["high"]
print(tails.assign(**{c: (tails[c] * 100).round(1) for c in ["low", "high", "both"]}).to_string())
