#Shared fixtures for all tests.

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="session")
def client():
    """TestClient with models loaded once for the whole test session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_features():
    """
    A valid input dict matching the CodeFeatures schema.
    Values are realistic JM1 numbers, not zeros, which can break scaler.
    """
    return {
        "loc": 50,
        "v(g)": 5,
        "ev(g)": 3,
        "iv(g)": 4,
        "n": 120,
        "v": 300.5,
        "l": 0.05,
        "d": 20.1,
        "i": 14.9,
        "e": 6040.5,
        "b": 0.1,
        "t": 335.6,
        "lOCode": 40,
        "lOComment": 5,
        "lOBlank": 5,
        "lOCodeAndComment": 2,
        "uniq_Op": 15,
        "uniq_Opnd": 20,
        "total_Op": 60,
        "total_Opnd": 60,
        "branchCount": 10,
    }


@pytest.fixture
def predict_payload(sample_features):
    return {"model": "decision_tree", "features": sample_features}