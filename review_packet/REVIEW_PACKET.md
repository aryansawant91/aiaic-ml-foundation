# REVIEW PACKET — AIAIC ML Foundation
**Candidate:** Aryan Sawant
**Test:** Test 1 — ML Foundation & Prediction Service
**Intake:** AIAIC 7-4-3
**Submitted:** August 2026
**Status:** COMPLETE

---

## Quick Summary

| Item | Result |
|------|--------|
| Real dataset used | ✅ 740,125 rows, Indian mandi price data |
| Deterministic preprocessing | ✅ Hash-verified at every stage |
| Feature engineering | ✅ Lag (1/7/14d), rolling (7/14/30d), seasonal |
| Model trained | ✅ LightGBM, R²=0.9946, MAPE=4.52% |
| Beats naive baseline | ✅ 10x better MAE (44 vs 435) |
| FastAPI service | ✅ /predict + /health, Pydantic validated |
| Docker deployment | ✅ Dockerfile + docker-compose (API + Mongo) |
| Unit tests | ✅ 42/42 passing |
| Replay-safe | ✅ SHA-256 content hash at every pipeline stage |
| Integration-ready | ✅ No modifications to existing architecture |

---

## Folder Index

| Folder/File | Contents |
|-------------|----------|
| `REVIEW_PACKET.md` | This file — master index |
| `Executive_Assessment.md` | Self-assessment of build quality |
| `Assignment_vs_Delivery.md` | Line-by-line requirement mapping |
| `architecture_summary.md` | System architecture explanation |
| `known_limitations.md` | Honest gaps and assumptions |
| `screenshots/` | API, pipeline, test, repo screenshots |
| `code_packet/` | 8 key files with explanations |
| `api_samples/` | predict + health JSON samples |
| `runtime_logs/` | Pipeline run log + test results |
| `deployment_proof/` | Docker setup proof |

---

## Key Numbers

- **740,125** rows ingested
- **737,389** clean rows after validation
- **2,736** rows quarantined (logged with reasons)
- **735,774** rows used for training after feature engineering
- **699,750** training rows / **36,024** test rows (time-based split)
- **R² 0.9946** — model explains 99.46% of price variance
- **MAE 44.07** INR/quintal average error
- **MAPE 4.52%** — production-quality accuracy
- **42/42** unit tests passing
- **~55 seconds** full pipeline runtime on 740k rows