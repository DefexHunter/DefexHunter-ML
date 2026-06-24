---
title: DefexHunter
emoji: 🐛
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: Predicts software defects from NASA JM1 static code metrics
---

# 🐛 DefexHunter ML

> Production-ready REST API for software defect prediction using the NASA JM1 dataset.

[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger-blue)](https://asmaatef-defexhunter.hf.space/docs)
[![Postman Collection](https://img.shields.io/badge/Postman-Collection-FF6C37)](https://documenter.getpostman.com/view/53206285/2sBXwwnnrT)

---

##  What is this?

DefexHunter predicts whether a software module is **defective or not** based on static code metrics (Halstead complexity, cyclomatic complexity, line counts). It trains 7 ML models on the NASA JM1 dataset and serves predictions via a FastAPI REST API.

---

##  Live API

```
Base URL: https://asmaatef-defexhunter.hf.space
Docs:     https://asmaatef-defexhunter.hf.space/docs
Health:   https://asmaatef-defexhunter.hf.space/health
Models:   https://asmaatef-defexhunter.hf.space/models
```

---

##  Models

| Model | Imbalance Strategy |
|---|---|
| XGBoost | scale_pos_weight + SMOTE |
| Random Forest | class_weight + SMOTE family / hybrid samplers |
| Decision Tree | class_weight + SMOTE |
| SVM | class_weight + SMOTE |
| Logistic Regression | class_weight + SMOTE |
| KNN | SMOTE |
| Naïve Bayes | SMOTE |

All models are tuned via **GridSearchCV** (5-fold StratifiedKFold, macro F1 scoring).

---

##  Pipeline

```
Raw CSV → Clean → Stratified Split (70/30) → Feature Selection → StandardScaler → Resampling → GridSearchCV
```

---

##  Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check; lists loaded models |
| GET | `/models` | Returns training metrics for all models |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Up to 100 predictions in one call |

---

##  Example Request

`POST /predict`

```json
{
  "features": {
    "loc": 50,
    "ev(g)": 3,
    "iv(g)": 5,
    "n": 120,
    "l": 0.4,
    "d": 12.5,
    "i": 8.0,
    "t": 450.0,
    "lOCode": 40,
    "lOComment": 5,
    "lOBlank": 5,
    "locCodeAndComment": 2,
    "uniq_Op": 15,
    "uniq_Opnd": 20,
    "branchCount": 10
  },
  "model": "xgboost"
}
```

Example response:

```json
{
  "prediction": 1,
  "label": "Defective",
  "probability": 0.87,
  "confidence": "High",
  "model_used": "xgboost"
}
```

---

##  Input Features

| Feature | Description |
|---|---|
| `loc` | Lines of code |
| `ev(g)` | Essential cyclomatic complexity |
| `iv(g)` | Design complexity |
| `n` | Halstead program length |
| `l` | Halstead volume ratio |
| `d` | Halstead difficulty |
| `i` | Halstead intelligence |
| `t` | Halstead programming time |
| `lOCode` | Lines of code only |
| `lOComment` | Lines of comment |
| `lOBlank` | Blank lines |
| `locCodeAndComment` | Code + comment lines |
| `uniq_Op` | Unique operators |
| `uniq_Opnd` | Unique operands |
| `branchCount` | Branch count |

---

##  Tech Stack

`FastAPI` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `pandas` · `NumPy` · `Docker` · `Git LFS`


##  Run Locally

```bash
# Train models first
python src/train.py data/jm1_csv.csv

# Run with Docker
docker-compose up --build

# Or directly
uvicorn main:app --host 0.0.0.0 --port 7860
```