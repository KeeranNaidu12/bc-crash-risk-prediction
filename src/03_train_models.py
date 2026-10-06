"""
03_train_models.py
Predict next year's crashes per intersection with Empirical Bayes (see empirical_bayes.py),
check it against simple baselines on past years, and save a 2026 forecast for Tableau.

Saves to dashboard/ (one row per ...):
  predictions.csv    intersection: 2026 forecast, 90% range, typical, excess, ranks
  crash_history.csv  intersection and year: crashes 2021-2025, plus the 2026 forecast
  validation.csv     validation/test year and method: MAE, deviance, top-100 share
  calibration.csv    validation/test year and decile of prediction: predicted vs actual

Intersections are ranked by EXCESS crashes (Empirical Bayes estimate minus typical):
how many more crashes a year an intersection has than is normal for its type. This
finds intersections that underperform, rather than just the busiest ones.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from empirical_bayes import (TEST_YEAR, add_history, coefficients, empirical_bayes, fit_spf,
                             forecast_rows, level_factor, load_raw, year_variation)

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"
FORECAST_YEAR = TEST_YEAR + 1

# 1-2. Crash counts and features from PostgreSQL, with history from previous years only
raw = load_raw()
df = add_history(raw)


# 3. Empirical Bayes: fit_spf, level_factor, year_variation and empirical_bayes live in
#    empirical_bayes.py
def top_k_crash_share(actual, predicted, k=100):
    """Share of the year's crashes that happened at the k intersections predicted riskiest."""
    return actual[predicted.nlargest(k).index].sum() / actual.sum()


# 4. Check on past years, one year at a time: fit on every year before Y, predict Y.
#    The regression is fitted on the history (years before Y) of the intersections
#    being predicted, never on their year-Y crashes.
#    2023 and 2024 are validation years (use them to compare methods and settings);
#    2025 is the final test and should only be looked at once changes are settled.
#    "Average x level" applies the same level factor to the plain average, so the
#    comparison with Empirical Bayes is not just about the level factor.
print(f"{'':16} {'':22} {'MAE':>5} {'Poisson dev':>11} {'top-100 share':>13} {'in 90% range':>12}")
validation, calibration = [], []
for year in range(df["year"].min() + 1, TEST_YEAR + 1):
    test = df[df["year"] == year].copy()
    level = level_factor(raw, test["location"], year)
    phi = year_variation(raw, test["location"], year)
    test = test.join(empirical_bayes(fit_spf(test), test, level=level, phi=phi))
    covered = test["crashes"].between(test["low_90"], test["high_90"]).mean()

    label = "TEST" if year == TEST_YEAR else "validation"
    for name, pred in [("Same as last year", test["prior_crashes"]),
                       ("Average of prior years", test["prior_avg"]),
                       ("Average x level", test["prior_avg"] * level),
                       ("Empirical Bayes", test["predicted"])]:
        mae = mean_absolute_error(test["crashes"], pred)
        deviance = mean_poisson_deviance(test["crashes"], pred.clip(lower=0.01))
        share = top_k_crash_share(test["crashes"], pred)
        coverage = f"{covered:12.1%}" if name == "Empirical Bayes" else ""
        print(f"{year} {label:10} {name:22} {mae:5.2f} {deviance:11.2f} {share:13.1%} {coverage}")
        validation.append({"year": year, "split": label.lower(), "method": name, "mae": mae,
                           "poisson_deviance": deviance, "top100_share": share,
                           "in_90_range": covered if name == "Empirical Bayes" else None})
    print(f"{year} {label:10} (level factor {level:.3f}, year-to-year variation {phi:.3f})")

    # Calibration: intersections in 10 equal groups by predicted crashes (ranked, as many
    # predictions tie at the low end), mean predicted vs mean actual in each
    decile = pd.qcut(test["predicted"].rank(method="first"), 10, labels=False) + 1
    calibration.append(test.groupby(decile)
                           .agg(intersections=("crashes", "size"),
                                mean_predicted=("predicted", "mean"), mean_actual=("crashes", "mean"))
                           .rename_axis("decile").reset_index().assign(year=year, split=label.lower()))

# 5. Forecast next year from all years of history
latest = forecast_rows(raw)
spf = fit_spf(latest)
level = level_factor(raw, latest["location"], FORECAST_YEAR)
phi = year_variation(raw, latest["location"], FORECAST_YEAR)
forecast = latest.join(empirical_bayes(spf, latest, level=level, phi=phi))
forecast["excess"] = forecast["predicted"] - forecast["typical"]

# 6. Explain the model: how each feature multiplies typical crashes per year
names = {"is_signalized": "signalized", "is_interchange": "interchange",
         "C(n_streets.clip(2, 4))[T.3]": "3 streets", "C(n_streets.clip(2, 4))[T.4]": "4+ streets"}
for street_class in ["arterial", "secondary_arterial", "collector", "none"]:
    names[f"C(street_class, Treatment('local'))[T.{street_class}]"] = street_class.replace("_", " ")
print("\nTypical crashes per year at an unsignalized 2-street intersection of local streets: "
      f"{np.exp(coefficients(spf)['Intercept']):.1f}")
for term, name in names.items():
    print(f"  {name:18} x{np.exp(coefficients(spf)[term]):.2f}")
print(f"Median weight on typical (vs the intersection's own history): {forecast['weight'].median():.2f}")
print(f"Level factor for {FORECAST_YEAR} (latest year's crashes over the average year's): {level:.3f}")

# 7. Save the forecast for Tableau, ranked by excess crashes. weight_on_typical is how much
#    the forecast relies on typical for its type rather than the intersection's own history.
output = forecast[["location", "latitude", "longitude", "street_class", "is_signalized",
                   "is_interchange", "n_streets", "prior_avg", "typical", "predicted", "low_90",
                   "high_90", "excess", "weight"]].copy()
output = output.rename(columns={"prior_avg": "avg_crashes_2021_2025", "typical": "typical_for_type",
                                "predicted": f"predicted_{FORECAST_YEAR}",
                                "weight": "weight_on_typical"})
# Round the crash figures for display; coordinates keep 6 decimals (~0.1 m) for mapping
output = output.round(1).assign(latitude=output["latitude"].round(6),
                                longitude=output["longitude"].round(6),
                                weight_on_typical=output["weight_on_typical"].round(3))
output["excess_rank"] = output["excess"].rank(ascending=False, method="first").astype(int)
output["predicted_rank"] = output[f"predicted_{FORECAST_YEAR}"].rank(ascending=False, method="first").astype(int)
output = output.sort_values("excess_rank")
output.to_csv(DASHBOARD / "predictions.csv", index=False)
print(f"\nTop 10 intersections by excess crashes ({FORECAST_YEAR} forecast):")
print(output.head(10)[["location", f"predicted_{FORECAST_YEAR}", "low_90", "high_90",
                       "typical_for_type", "excess"]].to_string(index=False))

# 8. Save crashes per intersection per year (long format, for trend lines), with the
#    forecast as one more year: kind is "actual" or "forecast"; the range is forecast only
actual = raw[["location", "year", "crashes"]].assign(kind="actual")
predicted = output[["location", f"predicted_{FORECAST_YEAR}", "low_90", "high_90"]].rename(
    columns={f"predicted_{FORECAST_YEAR}": "crashes"}).assign(year=FORECAST_YEAR, kind="forecast")
history = pd.concat([actual, predicted], ignore_index=True).sort_values(["location", "year"])
history = history[["location", "year", "kind", "crashes", "low_90", "high_90"]]
history.to_csv(DASHBOARD / "crash_history.csv", index=False)

# 9. Save the validation results and calibration (Empirical Bayes) for a model-quality panel
pd.DataFrame(validation).round(4).to_csv(DASHBOARD / "validation.csv", index=False)
calibration = pd.concat(calibration)[["year", "split", "decile", "intersections",
                                      "mean_predicted", "mean_actual"]]
calibration.round(3).to_csv(DASHBOARD / "calibration.csv", index=False)
print("\nSaved dashboard/predictions.csv, crash_history.csv, validation.csv, calibration.csv")
