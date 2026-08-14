# Known Limitations & Assumptions
**Service:** AIAIC ML Foundation | **Candidate:** Aryan Sawant

---

## Dataset Assumptions

**Date format**
`Agriculture_price_dataset.csv` uses `M/D/YYYY` format (e.g. `6/6/2023`).
`commodity_price.csv` uses a different format that failed parsing —
2,733 rows quarantined as `unparseable_date`. These are from the smaller
supplementary file and do not affect training quality meaningfully.

**Price units**
All prices assumed to be in INR per quintal. No unit normalization was applied.
If source data contains mixed units, predictions will be off for those rows.

**Commodity spelling**
Commodity names are matched exactly (case-sensitive after normalization).
"Onion" and "onion" would be treated as different categories.

---

## Pipeline Limitations

**Model reload requires restart**
The trained model is cached with `@lru_cache` on first API request.
If a new model is trained while the API is running, the API must be
restarted to pick it up. A future `/reload` endpoint would fix this.

**No MLflow tracking**
A lightweight JSON run registry is used instead of MLflow. Full MLflow
integration (experiment tracking, artifact store, UI) is not implemented
but the run registry schema is compatible with MLflow concepts.

**Agmarknet live API not connected**
The service was originally designed to pull live data from the
data.gov.in Agmarknet API. This was replaced with local Kaggle CSVs
because API key registration was blocked during development. The client
code exists in `src/ingestion/agmarknet_client.py` and can be activated
by setting `AIAIC_AGMARKNET_API_KEY` in `.env`.

---

## Deployment Limitations

**Docker not smoke-tested locally**
Docker Desktop was not available in the local build environment.
The Dockerfile and docker-compose.yml are production-ready and reviewed
for correctness, but were not run end-to-end locally. The API was fully
verified without Docker using uvicorn directly.

**MongoDB optional**
MongoDB is used for dataset version mirroring and training run logging.
If MongoDB is unreachable, the service degrades gracefully to local JSON.
This is by design — the service must work in environments without a
running MongoDB instance.

---

## Model Limitations

**Price scale varies by commodity**
A single model is trained across all commodities and states. Commodity-specific
models would likely improve accuracy for individual crops but were not built
in this version.

**No retraining trigger**
There is no automated retraining when new data arrives. Retraining requires
manually running `python run_pipeline.py` again.

**Feature availability at inference time**
The model's strongest features are lag and rolling price features
(modal_price_lag_1, modal_price_roll_mean_7 etc.). If a caller doesn't
supply these (e.g. for a completely new market with no price history),
prediction quality will degrade. The API logs a warning when features
are missing but still returns a prediction.