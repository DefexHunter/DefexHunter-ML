#Integration tests-  hits the FastAPI endpoints via TestClient.
#Models must be trained (models/*.pkl must exist) before running these.

import pytest


class TestHealth:
    def test_health_returns_ok(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_health_lists_models(self, client):
        res = client.get("/health")
        assert len(res.json()["models_loaded"]) > 0


class TestModels:
    def test_models_returns_list(self, client):
        res = client.get("/models")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_models_have_required_fields(self, client):
        res = client.get("/models")
        for m in res.json():
            assert "model"     in m
            assert "f1_score"  in m
            assert "accuracy"  in m
            assert "roc_auc"   in m


class TestPredict:
    def test_predict_returns_200(self, client, predict_payload):
        res = client.post("/predict", json=predict_payload)
        assert res.status_code == 200

    def test_predict_response_structure(self, client, predict_payload):
        res = client.post("/predict", json=predict_payload)
        body = res.json()
        assert "prediction"  in body
        assert "label"       in body
        assert "probability" in body
        assert "confidence"  in body
        assert "model_used"  in body

    def test_prediction_is_binary(self, client, predict_payload):
        res = client.post("/predict", json=predict_payload)
        assert res.json()["prediction"] in [0, 1]

    def test_label_matches_prediction(self, client, predict_payload):
        res = client.post("/predict", json=predict_payload)
        body = res.json()
        if body["prediction"] == 1:
            assert body["label"] == "Defective"
        else:
            assert body["label"] == "No Defect"

    def test_all_models_work(self, client, sample_features):
        for model_name in ["decision_tree", "knn", "random_forest", "svm", "xgboost"]:
            res = client.post("/predict", json={
                "model": model_name,
                "features": sample_features
            })
            assert res.status_code == 200, f"{model_name} failed: {res.text}"

    def test_unknown_model_returns_422(self, client, sample_features):
        res = client.post("/predict", json={
            "model": "nonexistent_model",
            "features": sample_features
        })
        assert res.status_code == 422

    def test_missing_feature_returns_422(self, client):
        res = client.post("/predict", json={
            "model": "decision_tree",
            "features": {"loc": 50}  # incomplete
        })
        assert res.status_code == 422

    def test_wrong_type_returns_422(self, client, sample_features):
        bad = sample_features.copy()
        bad["loc"] = "not_a_number"
        res = client.post("/predict", json={
            "model": "decision_tree",
            "features": bad
        })
        assert res.status_code == 422


class TestBatchPredict:
    def test_batch_returns_list(self, client, predict_payload):
        res = client.post("/predict/batch", json={"requests": [predict_payload, predict_payload]})
        assert res.status_code == 200
        assert len(res.json()) == 2

    def test_batch_over_100_returns_422(self, client, predict_payload):
        res = client.post("/predict/batch", json={
            "requests": [predict_payload] * 101
        })
        assert res.status_code == 422