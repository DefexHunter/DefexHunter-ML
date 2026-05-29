from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── input ─────────────────────────────────────────────────────────────────────
class CodeFeatures(BaseModel):
    """
    One software module's metrics.
    Field names = selected_features saved by train.py.
    All values are floats - raw numbers from static analysis tools.
    """
    loc:              float = Field(..., description="Lines of code")
    v_g:              float = Field(..., alias="v(g)",  description="McCabe cyclomatic complexity")
    ev_g:             float = Field(..., alias="ev(g)", description="McCabe essential complexity")
    iv_g:             float = Field(..., alias="iv(g)", description="McCabe design complexity")
    n:                float = Field(..., description="Halstead total N")
    v:                float = Field(..., description="Halstead volume")
    l:                float = Field(..., description="Halstead program length")
    d:                float = Field(..., description="Halstead difficulty")
    i:                float = Field(..., description="Halstead intelligence")
    e:                float = Field(..., description="Halstead effort")
    b:                float = Field(..., description="Halstead error estimate")
    t:                float = Field(..., description="Halstead time estimate")
    lOCode:           float = Field(..., description="Lines of code (Halstead)")
    lOComment:        float = Field(..., description="Lines of comment")
    lOBlank:          float = Field(..., description="Blank lines")
    lOCodeAndComment: float = Field(..., description="Lines of code and comment")
    uniq_Op:          float = Field(..., description="Unique operators")
    uniq_Opnd:        float = Field(..., description="Unique operands")
    total_Op:         float = Field(..., description="Total operators")
    total_Opnd:       float = Field(..., description="Total operands")
    branchCount:      float = Field(..., description="Branch count")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "loc": 50, "v(g)": 5, "ev(g)": 3, "iv(g)": 4,
                "n": 120, "v": 300.5, "l": 0.05, "d": 20.1,
                "i": 14.9, "e": 6040.5, "b": 0.1, "t": 335.6,
                "lOCode": 40, "lOComment": 5, "lOBlank": 5,
                "lOCodeAndComment": 2, "uniq_Op": 15, "uniq_Opnd": 20,
                "total_Op": 60, "total_Opnd": 60, "branchCount": 10,
            }
        },
    }


class PredictRequest(BaseModel):
    features: CodeFeatures
    model: Literal[
        "decision_tree", "knn", "random_forest", "svm", "xgboost"
    ] = Field(default="xgboost", description="Which trained model to use")


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