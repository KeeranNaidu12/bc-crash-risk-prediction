"""
03_train_models.py
Train LightGBM to predict crashes per intersection, compare it with a simple baseline,
explain it with SHAP, and save predictions for Tableau.
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
from db import get_engine

ROOT = Path(__file__).resolve().parents[1]
TEST_YEAR = 2025

FEATURES = ["prior_crashes", "prior_2yr_avg", "prior_casualty",
            "n_streets", "is_interchange", "is_signalized", "latitude", "longitude"]


# 1. Load crash counts and intersection features from PostgreSQL
df = pd.read_sql("""
    SELECT y.location, y.year, y.crashes, y.casualty_crashes,
           f.latitude, f.longitude, f.n_streets, f.is_interchange, f.is_signalized
    FROM intersection_year y
    JOIN intersection_features f USING (location)
""", get_engine())

# 2. History features: only use PREVIOUS years, so the model can't see the answer
df = df.sort_values(["location", "year"])
history = df.groupby("location")
df["prior_crashes"] = history["crashes"].shift(1)
df["prior_2yr_avg"] = history["crashes"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
df["prior_casualty"] = history["casualty_crashes"].shift(1)
df[["is_interchange", "is_signalized"]] = df[["is_interchange", "is_signalized"]].astype(int)
df = df.dropna(subset=["prior_crashes"])  # 2021 has no previous year

# ICBC only lists intersections that had a crash, so an intersection whose first crash
# is in year Y would only be in the data because of year Y. Keep a row only if the
# intersection had a crash in an EARLIER year, i.e. it was known when predicting.
df["known_before"] = history["crashes"].transform(lambda s: s.shift(1).cumsum()) > 0
df = df[df["known_before"]]

# 3. Split by time, one year at a time: train on every year before Y, predict Y.
#    2023 and 2024 are validation years (use them to compare features/settings);
#    2025 is the final test and should only be looked at once changes are settled.
def train_model(train):
    """LightGBM with a Poisson objective, since the target is a count."""
    model = lgb.LGBMRegressor(objective="poisson", n_estimators=400, learning_rate=0.03,
                              min_child_samples=30, random_state=42, verbose=-1)
    return model.fit(train[FEATURES], train["crashes"])


def top_k_hits(actual, predicted, k):
    """How many of the k intersections with the most crashes were also predicted in the top k."""
    return len(set(actual.nlargest(k).index) & set(predicted.nlargest(k).index))


# 4. Compare LightGBM with a simple baseline, "same as last year", in every year
print(f"{'':10} {'':22} {'MAE':>5} {'top-20':>7} {'top-100':>8}")
for year in range(df["year"].min() + 1, TEST_YEAR + 1):
    train = df[df["year"] < year]
    test = df[df["year"] == year].copy()
    model = train_model(train)
    test["predicted"] = model.predict(test[FEATURES])

    label = "TEST" if year == TEST_YEAR else "validation"
    for name, pred in [("Baseline (last year)", test["prior_crashes"]), ("LightGBM", test["predicted"])]:
        mae = mean_absolute_error(test["crashes"], pred)
        top20 = top_k_hits(test["crashes"], pred, 20)
        top100 = top_k_hits(test["crashes"], pred, 100)
        print(f"{year} {label:10} {name:22} {mae:5.2f} {top20:4}/20 {top100:4}/100")

# The loop ends on TEST_YEAR, so model and test are the final 2022-2024 model and its 2025 predictions

# 6. Explain the model with SHAP
shap_values = shap.TreeExplainer(model).shap_values(test[FEATURES])
shap.summary_plot(shap_values, test[FEATURES], show=False)
plt.tight_layout()
plt.savefig(ROOT / "images" / "shap_summary.png", dpi=150)
print("Saved images/shap_summary.png")

# 7. Save predictions for Tableau
output = test[["location", "latitude", "longitude", "crashes", "predicted"]].copy()
output = output.rename(columns={"crashes": "actual_2025", "predicted": "predicted_2025"})
output["predicted_2025"] = output["predicted_2025"].round(1)
output["risk_rank"] = output["predicted_2025"].rank(ascending=False, method="first").astype(int)
output.sort_values("risk_rank").to_csv(ROOT / "dashboard" / "predictions.csv", index=False)
print("Saved dashboard/predictions.csv")