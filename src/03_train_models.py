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

# 3. Split by time: learn from 2022-2024, test on 2025
train = df[df["year"] < TEST_YEAR]
test = df[df["year"] == TEST_YEAR].copy()

# 4. Train LightGBM (Poisson objective = predicting counts)
model = lgb.LGBMRegressor(objective="poisson", n_estimators=400, learning_rate=0.03,
                          min_child_samples=30, random_state=42, verbose=-1)
model.fit(train[FEATURES], train["crashes"])
test["predicted"] = model.predict(test[FEATURES])

# 5. Compare with a simple baseline: "same as last year"
def top20_hit_rate(actual, predicted):
    top_actual = set(actual.nlargest(20).index)
    top_predicted = set(predicted.nlargest(20).index)
    return len(top_actual & top_predicted) / 20

print(f"Results on {TEST_YEAR}:")
for name, pred in [("Baseline (last year)", test["prior_crashes"]), ("LightGBM", test["predicted"])]:
    mae = mean_absolute_error(test["crashes"], pred)
    hits = top20_hit_rate(test["crashes"], pred)
    print(f"  {name:22s} MAE = {mae:.2f}   top-20 hit rate = {hits:.0%}")

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