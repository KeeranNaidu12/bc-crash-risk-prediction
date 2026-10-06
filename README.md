# Vancouver Intersection Crash Risk Prediction

Predicting crash risk at Vancouver intersections using ICBC crash data and City of Vancouver open data.

## Setup

1. Install PostgreSQL and create a database and user.
2. Copy `.env.example` to `.env` and fill in your database login.
3. Create a virtual environment and install dependencies:
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```
4. Download the data into `data/raw/`:
	- **ICBC Reported Crashes** from ICBC's Tableau Public profile (Download → Tableau Workbook), saved as `ICBC_Reported_Crashes.twbx`
	- **Traffic signals** from the City of Vancouver open data portal, saved as `traffic-signals.geojson`
	- **Public streets** (street class of every block) from the City of Vancouver open data portal, saved as `public-streets.geojson`. `02_load_city_data.py` downloads both City files if they are missing.
5. Run the pipeline (in this order):
	```bash
	python src/01_load_icbc.py
	python src/02_load_city_data.py
	python src/03_train_models.py
	python src/04_evaluate_model.py   # optional: model diagnostics, printed to the terminal
	```

## Data pipeline

| Stage | Rows | Crashes |
|---|---|---|
| ICBC Reported Crashes, all of BC, 2021–2025 (`icbc_crashes`) | 1,453,136 | 1,465,435 |
| Vancouver only | 232,042 | 233,440 |
| Excluding parking-lot and parked-vehicle crashes (`vancouver_crashes`) | 128,616 | 129,537 |
| Intersection crashes only | 105,306 | 106,066 |
| Summed per intersection per year (`intersection_year`) | 22,210 (4,442 × 5) | 106,066 |

ICBC publishes some rows that stand for more than one crash (`total_crashes` > 1), so crashes are always counted with `SUM(total_crashes)`, never by counting rows. For the same reason the data is not de-duplicated: ICBC has already merged identical crashes, so two identical rows would be two real crashes.

`intersection_features` holds one row per intersection: number of streets, whether it is an interchange (bridge, ramp, highway or connector), whether a City traffic signal lies within 30 m, and its street class.

**Street class** is the most major City street class among the street blocks within 15 m of the intersection: arterial (1,322 intersections), secondary arterial (592), collector (218) or local (2,268). The City publishes no traffic counts per intersection, so street class stands in for traffic volume. 42 intersections have no City street block nearby (Stanley Park, Granville Island, the port and highway ramps, which the City does not classify) and get their own class, `none`. Results barely change with a 10 m or 25 m match distance.

### Data quality checks

- No duplicate rows, and no missing values in any column the model uses.
- Crash totals match between stages (106,066 intersection crashes in, 106,066 out).
- Every intersection has a single coordinate; no intersection appears under two names.
- `intersection_year.location` is a foreign key to `intersection_features`.

## Model

**Empirical Bayes** (the method in the Highway Safety Manual, used by road agencies to find high-risk sites) predicts each intersection's crashes for the next year. It blends two estimates:

1. **Typical crashes** for an intersection like this one, from a negative binomial regression on its features (street class, signalized, interchange, number of streets). In road-safety terms this is a safety performance function.
2. **The intersection's own history:** its average crashes per year so far.

The weight on the typical estimate is `1 / (1 + k × typical × years of history)`, where `k` is the regression's overdispersion. A busy intersection with several years of history is judged mostly on its own record. A quiet one is pulled towards what is typical for its type, which corrects for one-off bad or good years (regression to the mean).

**Zero-truncated regression.** ICBC only lists intersections that have had a crash, so every intersection with history has at least one crash in it. A regression fitted on those intersections alone already reflects that crash, and Empirical Bayes would then count it again from the history, over-predicting quiet intersections. Instead, the regression is fitted on each intersection's total crashes so far, allowing for the fact that totals of zero are never seen. It therefore estimates typical crashes for all intersections of a kind, including those that never crash. It is fitted only on history, never on the year being predicted.

**Level factor.** Crashes in Vancouver rose from 17,920 in 2021 to about 23,000 in 2024 and 2025, so an average over past years under-predicts the next one. The forecast is scaled so that, summed over the intersections predicted, it equals their crashes in the most recent year, which carries the latest level forward. This is the Highway Safety Manual's calibration factor. The most recent year is 7–11% above the average year in the validation and test years, and 8% above it for 2026. On the validation years, carrying the latest level forward beat both no adjustment and extending a straight-line trend; the straight line overshoots because crashes levelled off.

**Year-to-year variation.** Empirical Bayes treats each intersection's crash rate as fixed, but rates drift a little from year to year. The 90% ranges allow for this with a small extra variance (φ ≈ 0.01), estimated from how much each intersection's counts varied across past years.

Fitted on all years, typical crashes per year at an unsignalized two-street intersection of local streets are 0.4. They are ×9.9 where an arterial meets, ×6.0 for a secondary arterial and ×2.9 for a collector; ×4.3 at signalized intersections, ×4.2 at interchanges and ×4.7 where four or more streets meet. These are associations, not effects: busier streets get signals, so "signalized" partly stands in for traffic volume too.

### Validation

Validation is done one year at a time: the model predicts year Y using only the years before Y, as it would be used in practice. 2023 and 2024 are validation years for comparing methods and settings. 2025 is the final test year. Splits are always by year, never random, so the model never learns from a year later than the one it predicts.

| Year | Method | MAE | Poisson deviance | Top-100 share | Actual in 90% range |
|---|---|---|---|---|---|
| 2023 (validation) | Same as last year | 2.20 | 2.95 | 32.7% | |
| | Average of prior years | 1.94 | 1.59 | 32.8% | |
| | Average × level | 1.93 | 1.55 | 32.8% | |
| | Empirical Bayes | **1.89** | **1.46** | 32.8% | 95.4% |
| 2024 (validation) | Same as last year | 2.09 | 2.97 | 32.1% | |
| | Average of prior years | 1.88 | 1.60 | 32.2% | |
| | Average × level | 1.84 | 1.53 | 32.2% | |
| | Empirical Bayes | **1.83** | **1.49** | 32.2% | 93.7% |
| 2025 (test) | Same as last year | 1.95 | 2.91 | 31.1% | |
| | Average of prior years | **1.68** | 1.42 | 31.1% | |
| | Average × level | 1.69 | 1.38 | 31.1% | |
| | Empirical Bayes | **1.68** | **1.35** | 31.1% | 95.1% |

"Average × level" is the plain average scaled the same way, to the most recent year's level, so that the comparison is not just about the level factor.

Top-100 share is the share of the year's crashes that happened at the 100 intersections predicted riskiest. About 31–33% of crashes happen at the top 100 intersections, and every method picks nearly the same ones, so the riskiest intersections stay risky year after year.

All of them clearly beat "same as last year". Empirical Bayes has the lowest Poisson deviance and the lowest (or tied) MAE in every year. Deviance is the better measure for crash counts, because it scores the whole predicted distribution rather than a single number. In 2025, Empirical Bayes cuts deviance by 5% compared with the plain average (1.35 vs 1.42). Each change was chosen on 2023 and 2024 and added a share of that gain:

| Version | Deviance 2023 / 2024 |
|---|---|
| Empirical Bayes, signalized/interchange/streets only | 1.62 / 1.59 |
| + street class | 1.53 / 1.55 |
| + level factor | 1.50 / 1.49 |
| + zero-truncated regression and calibration | 1.46 / 1.49 |

Each Empirical Bayes prediction comes with a 90% range for the year's crash count. In every past year, 94–95% of actual counts fell inside their range.

### Diagnostics

`src/04_evaluate_model.py` checks the model on the same year-by-year splits and prints the results.

- **Optimization convergence.** The zero-truncated regression converges with BFGS in every fit. The gradient at the optimum is at most 0.025, and the Hessian is negative definite (condition number about 240). L-BFGS reaches the same log-likelihood (within 2×10⁻⁵) and coefficients (within 1.3×10⁻³). Newton's method fails on every fit, so BFGS is used. Overdispersion is `k` = 1.35 (95% CI 1.25–1.46).
- **Shrinkage.**
	- **The fitted `k` gives close to the right blend.** Scaling it by 0.75 is best in every year, but it lowers deviance by only 0.005 or less (1.456 vs 1.451 in 2023). So the model could trust typical very slightly more, but the gain is negligible.
	- **Shrinkage follows street class.** Local-street intersections lean on typical the most (median weight 0.32–0.48), arterials the least (0.01–0.02).
	- **The gain is largest where shrinkage is strongest.** Where the weight on typical is above 0.4, Empirical Bayes has 9% lower deviance than the average × level.
- **Calibration.**
	- **Totals carry the latest level forward.** Predicted totals are ×0.99, ×0.95 and ×1.02 of actual in 2023–2025. Without the level factor, Empirical Bayes under-predicted by 8–11% every year.
	- **Quiet intersections are no longer over-predicted.** In the lowest four deciles of prediction, actual over predicted is 1.00, 0.83, 0.99 and 1.02. Before the zero-truncated regression it was 0.79, 0.76, 0.71 and 0.87. At the 10% of intersections with the lowest average, Empirical Bayes predicts 0.56, 0.42 and 0.35 crashes against 0.54, 0.45 and 0.35 actual.
	- **Intersections with about 1–5 crashes a year are slightly under-predicted,** by about 8%. In the top decile actual is within 1% of predicted.
	- **The randomized PIT is close to flat,** with 9–11% of counts in every bin.
	- **Ranges hold up for every street class, except slightly at busy streets.** By street class, the share of counts outside the central 90% of the predicted distribution should be 10%. It is 9.7% for local streets, 10.4% for collectors, 11.5% for secondary arterials and 12.3% for arterials. The extra at arterials is mostly counts above the range (7.7% vs 5%), from years when crashes rose more than the level factor expected. The plain coverage table shows 97% at local streets only because counts are whole numbers: with a mean of 0.5 crashes, the narrowest range, 0–2, already holds about 98%.

### 2026 forecast and ranking

`dashboard/predictions.csv` is a forecast for 2026, using all five years of history for all 4,442 intersections. For each intersection it gives the predicted crashes, a 90% range, the typical crashes for its type, and the **excess**: predicted minus typical.

Intersections are ranked by excess (`excess_rank`), as in Highway Safety Manual network screening. Ranking by predicted crashes alone (`predicted_rank`, also included) mostly lists the busiest intersections. Ranking by excess asks which intersections have more crashes than is normal for their type, which is where engineering changes are most likely to help. For example, the unsignalized arterial intersections at Knight St & E 62nd Ave and Dunbar St & W 16th Ave are forecast about 30 crashes a year against a typical 4, which places them in the excess top 100 but not the predicted top 100.

At the top, the two rankings are similar: 17 of the top 20 and 92 of the top 100 are the same, as the intersections with the most crashes also have far more than typical. Over all intersections the rank correlation is 0.16 (0.44 before street class was added), so excess no longer mostly reflects how busy an intersection's streets are.

### Dashboard files

`03_train_models.py` writes four CSV files to `dashboard/` for Tableau:

| File | One row per | Use |
|---|---|---|
| `predictions.csv` | Intersection (4,442) | Map and rankings: 2026 forecast, 90% range, typical, excess, both ranks, features, and `weight_on_typical` (how much the forecast relies on typical for its type rather than the intersection's own history) |
| `crash_history.csv` | Intersection and year, 2021–2026 | Trend lines. `kind` is `actual` for 2021–2025 and `forecast` for 2026, which also has the 90% range |
| `validation.csv` | Year and method (2023–2025) | Model comparison: MAE, Poisson deviance, top-100 share, and 90% range coverage for Empirical Bayes |
| `calibration.csv` | Year and decile of prediction | Calibration plot: mean predicted vs mean actual crashes |

Join `crash_history.csv` to `predictions.csv` on `location`. In `predictions.csv`, latitude and longitude keep 6 decimal places; an earlier version rounded them to 1, which put every intersection on a few points.

### Why not LightGBM

LightGBM (Poisson objective, using last year's crashes, a two-year average, last year's casualty crashes, the intersection features and coordinates) was tried first and replaced. It was worse than the multi-year average in every year: MAE 2.45, 2.02 and 1.89 for 2023–2025, and a bootstrap 95% interval for the gap excluded zero in each year. With five years of history and a few static features, a flexible model fits noise that a simple average smooths out.

### Preprocessing

History uses only earlier years, and booleans are converted to 0/1. High-crash intersections are kept, not removed as outliers, since they are what the model is meant to find. The negative binomial regression models skewed, overdispersed counts directly, so the target is not log-transformed. Intersections with more than four streets are grouped with four, as there are too few to estimate separately, and the one location with a single street name is grouped with two-street intersections.

## Limitations

- **Only intersections with at least one crash are covered.** ICBC lists crashes, not intersections, so an intersection with no crashes in 2021–2025 is not in the data. Results describe how risky known crash locations are, not every intersection in the city.
- **Rows are only used once an intersection is known.** Because of the point above, an intersection whose first crash is in a given year would only be in the data because of that year's outcome. To avoid this leakage, a row is used for training or testing only if the intersection had a crash in an earlier year. This removes 265 intersections from the 2025 test set.
- **Mid-block crashes (18% of Vancouver road crashes) are not modelled.**
- **Parking-lot and parked-vehicle crashes are excluded,** following the definition ICBC uses for its crash maps.
- **No exact dates or times.** ICBC publishes year, month, day of week and a 3-hour window only, so the model predicts yearly counts.
- **Traffic signals and street classes are current, not historical.** The City files describe the network today, so an intersection that was signalized, or a street that was reclassified, during 2021–2025 is treated as it is now in every year.
- **Signal matching is by distance.** An intersection counts as signalized if a signal lies within 30 m of the point ICBC records. Most matches are under 10 m, but about 130 intersections fall between 10 and 50 m, and a large intersection could be misclassified.
- **`is_interchange` is a keyword rule** on the intersection name (BRIDGE, RAMP, HWY, CONN), not an official road classification.
- **Street class, not traffic volume.** Standard safety performance functions use traffic counts, the strongest predictor of crashes. The City publishes counts for only about 659 intersections, and not in a downloadable form, so street class stands in for them. It separates arterials from side streets but not a busy arterial from a quiet one, which limits how much better Empirical Bayes can do than a multi-year average.
- **Feature effects are associations.** For example, signalized intersections have more crashes because signals are placed at busy intersections, not because signals cause crashes.
- **Excess is measured against an approximate benchmark.** "Typical for its type" accounts for street class but not actual traffic, so a high excess can still partly reflect an unusually busy arterial, not poor design.
- **The zero-truncated regression extrapolates.** It infers how many intersections of each kind never crash from the shape of the negative binomial distribution, not from data on those intersections. If that shape is wrong, typical crashes, and so excess, are off, most of all for local streets.
- **Simple trend.** The level factor carries the latest citywide level forward. It assumes the next year looks like the most recent one, is the same for every intersection, and cannot anticipate a change in direction. In 2024, when crashes rose 7%, the forecast was 5% low.
- **Little validation data.** With five years of data there are just three years to validate on, and the 2025 test year was viewed several times during development.

## Data sources

Contains information licensed under ICBC's Open Data Licence. Traffic signal and street class data from the City of Vancouver Open Data Portal.
