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

LightGBM (Poisson objective) predicts each intersection's crash count for the next year from its crash history (last year, two-year average, last year's casualty crashes), its features, and its coordinates. It is compared with a "same as last year" baseline.

Validation is done one year at a time: the model is trained on every year before Y and predicts Y, as it would be used in practice. 2023 and 2024 are validation years for comparing features and settings. 2025 is the final test year. Splits are always by year, never random, so the model never learns from a year later than the one it predicts.

Preprocessing: history features use only earlier years, booleans are converted to 0/1, and rows are split by year, never randomly. Features are not scaled, because tree models split on thresholds and are unaffected by scale (tested: 2025 MAE 1.887 unscaled vs 1.884 scaled). High-crash intersections are kept, not removed as outliers, since they are what the model is meant to find. The Poisson objective handles the skewed crash counts, so the target is not log-transformed.

| Year | Trained on | Model | MAE | Top-20 | Top-100 |
|---|---|---|---|---|---|
| 2023 (validation) | 2022 | Baseline | **2.20** | 16 | 88 |
| | | LightGBM | 2.45 | **17** | **89** |
| 2024 (validation) | 2022–2023 | Baseline | 2.09 | **16** | **88** |
| | | LightGBM | **2.02** | 14 | 87 |
| 2025 (test) | 2022–2024 | Baseline | 1.95 | **16** | 85 |
| | | LightGBM | **1.89** | 15 | **86** |

Top-k counts how many of the k intersections with the most crashes that year were also in the predicted top k. LightGBM has a lower MAE than the baseline once it has at least two years of history, and the gap grows with more training years. On ranking the two are within one or two intersections every year, which is too small a difference to call either one better.

## Limitations

- **Only intersections with at least one crash are covered.** ICBC lists crashes, not intersections, so an intersection with no crashes in 2021–2025 is not in the data. Results describe how risky known crash locations are, not every intersection in the city.
- **Rows are only used once an intersection is known.** Because of the point above, an intersection whose first crash is in a given year would only be in the data because of that year's outcome. To avoid this leakage, a row is used for training or testing only if the intersection had a crash in an earlier year. This removes 265 intersections from the 2025 test set.
- **Mid-block crashes (18% of Vancouver road crashes) are not modelled.**
- **Parking-lot and parked-vehicle crashes are excluded,** following the definition ICBC uses for its crash maps.
- **No exact dates or times.** ICBC publishes year, month, day of week and a 3-hour window only, so the model predicts yearly counts.
- **Traffic signals are current, not historical.** The City file lists today's signals, so an intersection that was signalized during 2021–2025 is treated as signalized in every year.
- **Signal matching is by distance.** An intersection counts as signalized if a signal lies within 30 m of the point ICBC records. Most matches are under 10 m, but about 130 intersections fall between 10 and 50 m, and a large intersection could be misclassified.
- **`is_interchange` is a keyword rule** on the intersection name (BRIDGE, RAMP, HWY, CONN), not an official road classification.
- **The model does not beat the baseline at ranking.** It has a lower MAE in 2024 and 2025, but its top-20 and top-100 hit counts are within one or two of "same as last year". With only five years of data there are just three years to validate on.
- **Model settings were not tuned.** The LightGBM settings were set by hand, not chosen by searching on the validation years.

## Data sources

Contains information licensed under ICBC's Open Data Licence. Traffic signal data from the City of Vancouver Open Data Portal.
