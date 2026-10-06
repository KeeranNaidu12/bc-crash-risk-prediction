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
5. Run the pipeline (in this order):
	```bash
	python src/01_load_icbc.py
	python src/02_load_city_data.py
	python src/03_train_models.py
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

`intersection_features` holds one row per intersection: number of streets, whether it is an interchange (bridge, ramp, highway or connector), and whether a City traffic signal lies within 30 m.

### Data quality checks

- No duplicate rows, and no missing values in any column the model uses.
- Crash totals match between stages (106,066 intersection crashes in, 106,066 out).
- Every intersection has a single coordinate; no intersection appears under two names.
- `intersection_year.location` is a foreign key to `intersection_features`.

## Model

**Empirical Bayes** (the method in the Highway Safety Manual, used by road agencies to find high-risk sites) predicts each intersection's crashes for the next year. It blends two estimates:

1. **Typical crashes** for an intersection like this one, from a negative binomial regression on its features (signalized, interchange, number of streets). In road-safety terms this is a safety performance function.
2. **The intersection's own history:** its average crashes per year so far.

The weight on the typical estimate is `1 / (1 + k × typical × years of history)`, where `k` is the regression's overdispersion. A busy intersection with several years of history is judged mostly on its own record. A quiet one is pulled towards what is typical for its type, which corrects for one-off bad or good years (regression to the mean).

Fitted on all years, typical crashes per year at an unsignalized two-street intersection are 2.0. They are ×7.8 at signalized intersections, ×3.0 at interchanges and ×7.6 where four or more streets meet. These are associations, not effects: signals are installed at busy intersections, so "signalized" mostly stands in for traffic volume.

### Validation

Validation is done one year at a time: the model is fit on every year before Y and predicts Y, as it would be used in practice. 2023 and 2024 are validation years for comparing methods and settings. 2025 is the final test year. Splits are always by year, never random, so the model never learns from a year later than the one it predicts.

| Year | Method | MAE | Poisson deviance | Top-100 share | Actual in 90% range |
|---|---|---|---|---|---|
| 2023 (validation) | Same as last year | 2.20 | 2.95 | 32.7% | |
| | Average of prior years | **1.94** | **1.59** | 32.8% | |
| | Empirical Bayes | 2.00 | 1.62 | 32.8% | 94.5% |
| 2024 (validation) | Same as last year | 2.09 | 2.97 | 32.1% | |
| | Average of prior years | **1.88** | 1.60 | 32.2% | |
| | Empirical Bayes | 1.91 | **1.59** | 32.2% | 92.9% |
| 2025 (test) | Same as last year | 1.95 | 2.91 | 31.1% | |
| | Average of prior years | **1.68** | 1.42 | 31.1% | |
| | Empirical Bayes | 1.71 | 1.42 | 31.1% | 93.7% |

Top-100 share is the share of the year's crashes that happened at the 100 intersections predicted riskiest. About 31–33% of crashes happen at the top 100 intersections, and every method picks nearly the same ones, so the riskiest intersections stay risky year after year.

Empirical Bayes and the multi-year average are effectively tied, and both clearly beat "same as last year". Empirical Bayes gives the typical estimate a median weight of only 0.06–0.08, because without traffic volume the regression cannot tell a quiet side street from a busy arterial with the same features. So it relies mostly on each intersection's own history. Its advantage over a plain average would grow with traffic volume data.

Each Empirical Bayes prediction comes with a 90% range for the year's crash count, from the negative binomial distribution the method implies. In every past year, 93–95% of actual counts fell inside their range, so the ranges are reliable and slightly conservative.

### 2026 forecast and ranking

`dashboard/predictions.csv` is a forecast for 2026, using all five years of history for all 4,442 intersections. For each intersection it gives the predicted crashes, a 90% range, the typical crashes for its type, and the **excess**: predicted minus typical.

Intersections are ranked by excess (`excess_rank`), as in Highway Safety Manual network screening. Ranking by predicted crashes alone (`predicted_rank`, also included) mostly lists the busiest intersections. Ranking by excess asks which intersections have more crashes than is normal for their type, which is where engineering changes are most likely to help. For example, the unsignalized intersections at Knight St & E 62nd Ave and Dunbar St & W 16th Ave are forecast about 27 crashes a year against a typical 2, which places them in the excess top 100 but not the predicted top 100.

Without traffic volume the two rankings are similar at the top: 18 of the top 20 and 93 of the top 100 are the same. The rank correlation over all intersections is 0.44.

### Why not LightGBM

LightGBM (Poisson objective, using last year's crashes, a two-year average, last year's casualty crashes, the intersection features and coordinates) was tried first and replaced. It was worse than the multi-year average in every year: MAE 2.45, 2.02 and 1.89 for 2023–2025, and a bootstrap 95% interval for the gap excluded zero in each year. With five years of history and a few static features, a flexible model fits noise that a simple average smooths out.

### Preprocessing

History features use only earlier years, and booleans are converted to 0/1. High-crash intersections are kept, not removed as outliers, since they are what the model is meant to find. The negative binomial regression models skewed, overdispersed counts directly, so the target is not log-transformed. Intersections with more than four streets are grouped with four, as there are too few to estimate separately, and the one location with a single street name is grouped with two-street intersections.

## Limitations

- **Only intersections with at least one crash are covered.** ICBC lists crashes, not intersections, so an intersection with no crashes in 2021–2025 is not in the data. Results describe how risky known crash locations are, not every intersection in the city.
- **Rows are only used once an intersection is known.** Because of the point above, an intersection whose first crash is in a given year would only be in the data because of that year's outcome. To avoid this leakage, a row is used for training or testing only if the intersection had a crash in an earlier year. This removes 265 intersections from the 2025 test set.
- **Mid-block crashes (18% of Vancouver road crashes) are not modelled.**
- **Parking-lot and parked-vehicle crashes are excluded,** following the definition ICBC uses for its crash maps.
- **No exact dates or times.** ICBC publishes year, month, day of week and a 3-hour window only, so the model predicts yearly counts.
- **Traffic signals are current, not historical.** The City file lists today's signals, so an intersection that was signalized during 2021–2025 is treated as signalized in every year.
- **Signal matching is by distance.** An intersection counts as signalized if a signal lies within 30 m of the point ICBC records. Most matches are under 10 m, but about 130 intersections fall between 10 and 50 m, and a large intersection could be misclassified.
- **`is_interchange` is a keyword rule** on the intersection name (BRIDGE, RAMP, HWY, CONN), not an official road classification.
- **No traffic volume.** Standard safety performance functions use traffic volume, the strongest predictor of crashes. Without it the regression is weak, and Empirical Bayes performs about the same as a multi-year average.
- **Feature effects are associations.** For example, signalized intersections have more crashes because signals are placed at busy intersections, not because signals cause crashes.
- **Excess is measured against a weak benchmark.** Without traffic volume, "typical for its type" cannot account for how busy an intersection is, so a high excess can still partly reflect heavy traffic, not poor design.
- **No trend over time.** The forecast assumes crash rates stay as they were in 2021–2025 and does not model year-to-year changes in overall crash levels.
- **Little validation data.** With five years of data there are just three years to validate on, and the 2025 test year was viewed several times during development.

## Data sources

Contains information licensed under ICBC's Open Data Licence. Traffic signal data from the City of Vancouver Open Data Portal.
