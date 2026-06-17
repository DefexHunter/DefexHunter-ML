import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import NearMiss

from data.config import (
    TEST_SIZE,
    RANDOM_STATE,
    CORRELATION_THRESHOLD,
    TARGET_COLUMN,
    MAX_MAJORITY_SAMPLES,
    MAX_MINORITY_SAMPLES,
    ENABLE_LOW_VARIANCE_FILTER,
    NZV_THRESHOLD,
    ENABLE_CONTRADICTION_REMOVAL,
    CONTRADICTION_THRESHOLD,
)


# -------------------- LOAD --------------------
def load_data(path):
    dataset = pd.read_csv(path)
    print("Shape:", dataset.shape)
    print("----------------Finished loading data----------------")
    return dataset


# -------------------- CLEAN --------------------
def clean_data(dataset, target=TARGET_COLUMN):
    dataset = dataset.copy()
    dataset[target] = dataset[target].astype(float)
    dataset.dropna(axis=0, inplace=True)
    print("Shape after cleaning:", dataset.shape)
    print("----------------Finished cleaning data----------------")
    return dataset


# -------------------- SPLIT --------------------
def split_data(dataset, target_col=TARGET_COLUMN):
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    print("----------------Finished splitting data----------------")
    return X_train, X_test, y_train, y_test


# -------------------- LOW VARIANCE (TRAIN-DERIVED) --------------------
def remove_low_variance_features(X_train, X_test, threshold=NZV_THRESHOLD):
    X_train = X_train.copy()
    X_test = X_test.copy()

    rng = (X_train.max() - X_train.min()).replace(0, 1e-9)
    normalized = (X_train - X_train.min()) / rng
    variances = normalized.var()

    keep_cols = variances[variances > threshold].index.tolist()
    dropped = [c for c in X_train.columns if c not in keep_cols]

    print(f"Near-zero-variance features dropped: {len(dropped)}")
    print("----------------Finished low-variance filtering----------------")
    return X_train[keep_cols], X_test[keep_cols], dropped


# -------------------- CORRELATED FEATURES (TRAIN ONLY) --------------------
def remove_correlated_features(
    X_train, X_test, y_train, threshold=CORRELATION_THRESHOLD, target_name=TARGET_COLUMN
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    corr = X_train.corr(numeric_only=True)
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], r))

    print(f"Highly correlated pairs: {len(high_corr_pairs)}")

    temp = X_train.copy().reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    temp[target_name] = y_train
    target_corr = temp.corr(numeric_only=True)[target_name].abs()

    to_drop = set()
    for a, b, _ in high_corr_pairs:
        if target_corr[a] >= target_corr[b]:
            to_drop.add(b)
        else:
            to_drop.add(a)

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)
    selected_features = sorted(X_train.columns.tolist())

    print(f"Dropped features: {len(to_drop)}")
    print(f"Selected features: {len(selected_features)}")
    print("----------------Finished removing correlated features----------------")

    return X_train, X_test, y_train, list(to_drop), selected_features


# -------------------- CONTRADICTORY SAMPLES (TRAIN ONLY) --------------------
def remove_contradictory_samples(X_train, y_train, threshold=CONTRADICTION_THRESHOLD):
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    nearest_idx = indices[:, 1]
    nearest_dist = distances[:, 1]
    y_arr = y_train.values

    contradictory = (y_arr[nearest_idx] != y_arr) & (nearest_dist <= threshold)
    keep_mask = ~contradictory

    print(f"Contradictory training samples removed: {int(contradictory.sum())} / {len(y_arr)}")
    print("----------------Finished contradictory-sample removal----------------")

    return (
        X_train.loc[keep_mask].reset_index(drop=True),
        y_train.loc[keep_mask].reset_index(drop=True),
    )


# -------------------- BALANCE (TRAIN ONLY) --------------------
def balance_data(
    X_train, y_train, max_majority_samples=MAX_MAJORITY_SAMPLES, max_minority_samples=MAX_MINORITY_SAMPLES
):
    class_counts = y_train.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_n = min(int(class_counts.max()), max_majority_samples)
    minority_n = min(int(class_counts.min()), max_minority_samples)

    print(f"Resampling → majority={majority_n}, minority={minority_n}")

    sampler = NearMiss(
        version=1,
        sampling_strategy={majority_class: majority_n, minority_class: minority_n},
    )

    X_res, y_res = sampler.fit_resample(
        X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    )

    print("After balancing:")
    print(pd.Series(y_res).value_counts())
    print("----------------Finished balancing data----------------")
    return X_res, y_res


# -------------------- SCALE --------------------
def scale_data(X_train, X_test):
    scaler = RobustScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print("----------------Finished scaling data----------------")
    return X_train_scaled, X_test_scaled, scaler


# -------------------- PIPELINE --------------------
def build_pipeline(path):
    data = load_data(path)
    data = clean_data(data)

    # Stratified split
    X_train, X_test, y_train, y_test = split_data(data)

    all_dropped = []

    # Correlation filtering (train only)
    X_train, X_test, y_train, dropped_corr, selected_features = (
        remove_correlated_features(
            X_train,
            X_test,
            y_train
        )
    )

    all_dropped.extend(dropped_corr)

    # Reset indexes
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )

    selected_features = sorted(X_train_scaled.columns.tolist())

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "selected_features": selected_features,
        "dropped_features": all_dropped,
    }