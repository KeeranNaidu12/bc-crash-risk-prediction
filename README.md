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
5. Run the pipeline:
	```bash
	python src/01_load_icbc.py
	python src/02_load_city_data.py
	```

## Data sources

Contains information licensed under ICBC's Open Data Licence. Traffic signal data from the City of Vancouver Open Data Portal.
