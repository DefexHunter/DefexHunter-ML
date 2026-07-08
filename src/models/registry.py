# src/models/registry.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

RANDOM_STATE = 42

# shared resampler choice: try "no resampling" (class_weight only) vs SMOTE
_RESAMPLER_CHOICES = ["passthrough", SMOTE(random_state=RANDOM_STATE)]


def _pipe(clf):
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("resampler", "passthrough"), 
            ("clf", clf),
        ]
    )



def _pipe_combo(clf):
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("over", "passthrough"),   # oversampling step (SMOTE family)
            ("under", "passthrough"),  # undersampling step (RandomUnderSampler)
            ("clf", clf),
        ]
    )


_RF_PARAM_GRID = [
    # 1) no resampling at all 
    {
        "over":  ["passthrough"],
        "under": ["passthrough"],
        "clf__n_estimators":      [200, 300, 500],
        "clf__max_depth":         [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2"],
        "clf__class_weight":      ["balanced", "balanced_subsample", None],
    },
    # 2) explicit over+under combo: SMOTE the minority up partway,
    #    then randomly undersample the majority down partway.
    #    sampling_strategy = desired minority:majority ratio after the step.
    {
        "over": [
            SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE),
            SMOTE(sampling_strategy=0.7, random_state=RANDOM_STATE),
            SMOTE(sampling_strategy=1.0, random_state=RANDOM_STATE),
            BorderlineSMOTE(sampling_strategy=0.7, random_state=RANDOM_STATE),
            ADASYN(sampling_strategy=0.7, random_state=RANDOM_STATE),
        ],
        "under": [
            "passthrough",
            RandomUnderSampler(sampling_strategy=0.8, random_state=RANDOM_STATE),
            RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_STATE),
        ],
        "clf__n_estimators":      [200, 300, 500],
        "clf__max_depth":         [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2"],
        "clf__class_weight":      [None, "balanced"],
    },
    # 3) hybrid samplers that do over+under in a single coordinated step
    {
        "over": [
            SMOTEENN(random_state=RANDOM_STATE),
            SMOTETomek(random_state=RANDOM_STATE),
        ],
        "under": ["passthrough"],
        "clf__n_estimators":      [200, 300, 500],
        "clf__max_depth":         [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__max_features":      ["sqrt", "log2"],
        "clf__class_weight":      [None, "balanced"],
    },
]


MODEL_REGISTRY = {

    "random_forest": (
        _pipe_combo(RandomForestClassifier(random_state=RANDOM_STATE)),
        _RF_PARAM_GRID,
    ),


    "decision_tree": (
         _pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
         {
             "resampler":            _RESAMPLER_CHOICES,
             "clf__max_depth":         [None, 5, 10, 20],
             "clf__min_samples_split": [2, 5, 10],
             "clf__min_samples_leaf":  [1, 2, 4],
             "clf__criterion":         ["gini", "entropy"],
             "clf__class_weight":      ["balanced", None],
         },
     ),

     "knn": (
         _pipe(KNeighborsClassifier()),
         {
             # kNN has no class_weight param -> resampling is its only lever
             "resampler":         [SMOTE(random_state=RANDOM_STATE)],
             "clf__n_neighbors": [3, 5, 7, 9, 11],
             "clf__weights":     ["uniform", "distance"],
             "clf__metric":      ["euclidean", "manhattan"],
         },
     ),

     "svm": (
         _pipe(SVC(probability=True, random_state=RANDOM_STATE)),
         [
             {
                 "resampler":       _RESAMPLER_CHOICES,
                 "clf__kernel":      ["linear"],
                 "clf__C":           [0.01, 0.1, 1, 10, 100],
                 "clf__class_weight": ["balanced", None],
             },
             {
                 "resampler":       _RESAMPLER_CHOICES,
                 "clf__kernel":      ["rbf"],
                 "clf__C":           [0.01, 0.1, 1, 10, 100],
                 "clf__gamma":       ["scale", "auto", 0.001, 0.01, 0.1, 1],
                 "clf__class_weight": ["balanced", None],
             },
         ],
     ),

     "xgboost": (
         _pipe(XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)),
         {
             "resampler":            ["passthrough", SMOTE(random_state=RANDOM_STATE)],
             "clf__n_estimators":     [100, 200],
             "clf__max_depth":        [3, 5, 7],
             "clf__learning_rate":    [0.01, 0.1],
             "clf__subsample":        [0.8, 1],
             "clf__colsample_bytree": [0.8, 1],
             # xgboost's class_weight equivalent: ratio of negative/positive
             "clf__scale_pos_weight": [1, 3, 5],
         },
     ),

     "logistic_regression": (
         _pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
         {
             "resampler":         _RESAMPLER_CHOICES,
             "clf__C":             [0.01, 0.1, 1, 10, 100],
             "clf__penalty":       ["l2"],
             "clf__class_weight":  ["balanced", None],
         },
     ),

     "naive_bayes": (
         _pipe(GaussianNB()),
         {
             # GaussianNB has no class_weight -> resampling is its only lever
             "resampler":         _RESAMPLER_CHOICES,
             "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
         },
     ),
}