from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── input ─────────────────────────────────────────────────────────────────────
class CodeFeatures(BaseModel):
    loc: float
    ev_g: float = Field(..., alias="ev(g)")
    iv_g: float = Field(..., alias="iv(g)")
    n: float
    l: float
    d: float
    i: float
    t: float

    lOCode: float
    lOComment: float
    lOBlank: float
    locCodeAndComment: float
    uniq_Op: float
    uniq_Opnd: float
    branchCount: float

    model_config = {
        "populate_by_name": True
    }

class PredictRequest(BaseModel):
    features: CodeFeatures
    model: Literal[
        "decision_tree", "knn", "random_forest", "svm", "xgboost",
        "logistic_regression", "naive_bayes"
    ] = Field(default="random_forest", description="Which trained model to use")


# ── output ────────────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    prediction:  int            = Field(..., description="0 = No Defect, 1 = Defective")
    label:       str            = Field(..., description="Human-readable label")
    probability: Optional[float] = Field(None, description="Confidence score 0–1")
    confidence:  str            = Field(..., description="High / Medium / Low")
    model_used:  str


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., max_length=100)


class ModelInfo(BaseModel):
    model:       str
    accuracy:    float
    cv_accuracy: float
    precision:   float
    recall:      float
    f1_score:    float
    roc_auc:     float
    class_0_acc: float
    class_1_acc: float