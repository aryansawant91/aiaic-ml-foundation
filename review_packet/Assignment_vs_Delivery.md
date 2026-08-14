# Assignment vs Delivery
**Candidate:** Aryan Sawant | **Test 1:** ML Foundation

---

| Requirement | Delivered | Evidence |
|-------------|-----------|----------|
| Real Dataset Loader | ✅ `src/ingestion/csv_loader.py` + `pipeline.py` | 740k rows loaded from 2 CSVs |
| Dataset Validation | ✅ `src/validation/pipeline.py` | 737k clean, 2.7k quarantined with reasons |
| Missing value handling | ✅ `src/preprocessing/missing_values.py` | Group-median fill, constant placeholders |
| Outlier handling | ✅ `src/preprocessing/outliers.py` | MAD-based robust z-score capping per group |
| Feature engineering | ✅ `src/features/` (4 files) | Lag 1/7/14d, rolling 7/14/30d, season encoding |
| Dataset version tracking | ✅ `src/ingestion/dataset_registry.py` | SHA-256 hash chain at every stage |
| Training pipeline | ✅ `src/training/train.py` | Time-based split, LightGBM, baseline comparison |
| Model persistence | ✅ `src/training/persistence.py` | .joblib + .json metadata, versioned by timestamp+hash |
| Prediction API | ✅ `src/api/routes.py` + `main.py` | POST /predict, GET /health |
| Inference pipeline | ✅ `src/inference/predictor.py` | Cached model load, feature row builder |
| Configuration management | ✅ `src/config/settings.py` | Fully env-driven, no magic numbers |
| Training logs | ✅ `evidence_packet/runtime_logs/pipeline_run.txt` | All 5 stages logged |
| Evaluation metrics | ✅ MAE/RMSE/MAPE/R² + baseline | Printed in pipeline run, saved in run registry |
| Sample predictions | ✅ `evidence_packet/api_samples/` | predict_response.json |
| Failure handling | ✅ Quarantine system + 503/422 API errors | ValidationError, PredictionError |
| Input validation | ✅ Pydantic schemas in `src/api/schemas.py` | 422 on missing required fields |
| Performance measurements | ✅ Pipeline report JSON | 55s runtime on 740k rows |
| Unit tests | ✅ `tests/` (8 test files) | 42/42 passing |
| README | ✅ `README.md` | Setup, run, API, integration notes |
| Architecture summary | ✅ `review_packet/architecture_summary.md` | Full system diagram |
| API documentation | ✅ Swagger UI at /docs + api_samples/ | Auto-generated + manual samples |
| Training instructions | ✅ README — Running the Pipeline section | Step by step |
| Dataset assumptions | ✅ `review_packet/known_limitations.md` | Documented |
| Known limitations | ✅ `review_packet/known_limitations.md` | Honest gaps listed |
| Dockerfile | ✅ `docker/Dockerfile` | Python 3.11-slim, non-root, healthcheck |
| Docker Compose | ✅ `docker/docker-compose.yml` | API + MongoDB, volumes, health dependency |
| Environment variables | ✅ `.env.example` | Documented with descriptions |
| Startup instructions | ✅ README — Docker Deployment section | Step by step |
| Health endpoint | ✅ `GET /health` | Returns model_loaded, mongo_available, status |
| Repository branch | ✅ github.com/aryansawant91/aiaic-ml-foundation | main branch |
| Screenshots | ✅ `evidence_packet/screenshots/` | API, pipeline, tests, repo |
| REVIEW_PACKET | ✅ `review_packet/` | All required files present |
| Evidence Packet | ✅ `evidence_packet/` | All required folders populated |