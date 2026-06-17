from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── input ─────────────────────────────────────────────────────────────────────
class CodeFeatures(BaseModel):
    """
    Matches the 22 features selected by the correlation-filter step in
    src/data/pipeline.py — see models/selected_features.json, produced by
    the most recent `python src/train.py` run:

    ['CALL_PAIRS', 'CYCLOMATIC_DENSITY', 'DECISION_COUNT', 'DESIGN_COMPLEXITY',
     'DESIGN_DENSITY', 'EDGE_COUNT', 'ESSENTIAL_COMPLEXITY', 'ESSENTIAL_DENSITY',
     'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY', 'HALSTEAD_EFFORT', 'HALSTEAD_LENGTH',
     'HALSTEAD_LEVEL', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS', 'MAINTENANCE_SEVERITY',
     'NORMALIZED_CYLOMATIC_COMPLEXITY', 'NUMBER_OF_LINES', 'NUM_UNIQUE_OPERANDS',
     'NUM_UNIQUE_OPERATORS', 'PARAMETER_COUNT', 'PERCENT_COMMENTS']

    Field names are snake_case for ergonomics; the `alias` on each field is the
    EXACT uppercase column name the model was trained on (typo included —
    "NORMALIZED_CYLOMATIC_COMPLEXITY" is how it appears in the dataset).
    predictor.predict() looks features up by that exact string, so the alias
    must never drift from selected_features.json.
    """

    call_pairs: float = Field(..., alias="CALL_PAIRS")
    cyclomatic_density: float = Field(..., alias="CYCLOMATIC_DENSITY")
    decision_count: float = Field(..., alias="DECISION_COUNT")
    design_complexity: float = Field(..., alias="DESIGN_COMPLEXITY")
    design_density: float = Field(..., alias="DESIGN_DENSITY")
    edge_count: float = Field(..., alias="EDGE_COUNT")
    essential_complexity: float = Field(..., alias="ESSENTIAL_COMPLEXITY")
    essential_density: float = Field(..., alias="ESSENTIAL_DENSITY")
    halstead_content: float = Field(..., alias="HALSTEAD_CONTENT")
    halstead_difficulty: float = Field(..., alias="HALSTEAD_DIFFICULTY")
    halstead_effort: float = Field(..., alias="HALSTEAD_EFFORT")
    halstead_length: float = Field(..., alias="HALSTEAD_LENGTH")
    halstead_level: float = Field(..., alias="HALSTEAD_LEVEL")
    loc_code_and_comment: float = Field(..., alias="LOC_CODE_AND_COMMENT")
    loc_comments: float = Field(..., alias="LOC_COMMENTS")
    maintenance_severity: float = Field(..., alias="MAINTENANCE_SEVERITY")
    normalized_cyclomatic_complexity: float = Field(
        ..., alias="NORMALIZED_CYLOMATIC_COMPLEXITY"
    )
    number_of_lines: float = Field(..., alias="NUMBER_OF_LINES")
    num_unique_operands: float = Field(..., alias="NUM_UNIQUE_OPERANDS")
    num_unique_operators: float = Field(..., alias="NUM_UNIQUE_OPERATORS")
    parameter_count: float = Field(..., alias="PARAMETER_COUNT")
    percent_comments: float = Field(..., alias="PERCENT_COMMENTS")

    model_config = {
        "populate_by_name": True
    }


class PredictRequest(BaseModel):
    features: CodeFeatures
    # Keep this in sync with whatever MODEL_REGISTRY in src/models/registry.py
    # actually produces a .pkl for. As of the latest train.py run + lightgbm
    # being added to requirements.txt: decision_tree, extra_trees, knn,
    # random_forest, svm, xgboost, lightgbm.
    # NOTE: "gradient_boosting" is deliberately excluded — its .pkl currently
    # fails to load (sklearn version mismatch between train and serve
    # environments, see predictor.py load_artifacts logs). Add it back once
    # that's resolved and the model is actually in _models.
    model: Literal[
        "decision_tree", "extra_trees", "knn", "random_forest",
        "svm", "xgboost", "lightgbm",
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