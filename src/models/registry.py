from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


MODEL_REGISTRY = {
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth":         [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "criterion":         ["gini", "entropy"],
        },
    ),
    "knn": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2"],
        },
    ),
    "svm": (
        SVC(probability=True, random_state=42),
        [
            {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10, 100]},
            {
                "kernel": ["rbf"],
                "C":      [0.01, 0.1, 1, 10, 100],
                "gamma":  ["scale", "auto", 0.001, 0.01, 0.1, 1],
            },
        ],
    ),
    "xgboost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {
            "n_estimators":     [100, 200],
            "max_depth":        [3, 5, 7],
            "learning_rate":    [0.01, 0.1],
            "subsample":        [0.8, 1],
            "colsample_bytree": [0.8, 1],
        },
    ),
}