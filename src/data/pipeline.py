import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss

from data.config import (
    TEST_SIZE,
    RANDOM_STATE,
    CORRELATION_THRESHOLD,
    TARGET_COLUMN,
    MAX_MAJORITY_SAMPLES,
    MAX_MINORITY_SAMPLES
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
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    print("----------------Finished splitting data----------------")

    return X_train, X_test, y_train, y_test


# -------------------- CORRELATED FEATURES (TRAIN ONLY) --------------------
def remove_correlated_features(
    X_train,
    X_test,
    y_train,
    threshold=CORRELATION_THRESHOLD,
    target_name=TARGET_COLUMN
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

    # safe alignment
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

    selected_features = list(X_train.columns)

    print(f"Dropped features: {len(to_drop)}")
    print(f"Selected features: {len(selected_features)}")
    print("----------------Finished removing correlated features----------------")

    return X_train, X_test, y_train, list(to_drop), selected_features


# -------------------- BALANCE (TRAIN ONLY) --------------------
def balance_data(
    X_train,
    y_train,
    max_majority_samples=MAX_MAJORITY_SAMPLES,
    max_minority_samples=MAX_MINORITY_SAMPLES
):

    class_counts = y_train.value_counts()

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_n = min(int(class_counts.max()), max_majority_samples)
    minority_n = min(int(class_counts.min()), max_minority_samples)

    print(f"Resampling → majority={majority_n}, minority={minority_n}")

    sampler = NearMiss(
        version=1,
        sampling_strategy={
            majority_class: majority_n,
            minority_class: minority_n
        }
    )

    X_res, y_res = sampler.fit_resample(X_train, y_train)

    print("After balancing:")
    print(pd.Series(y_res).value_counts())
    print("----------------Finished balancing data----------------")

    return X_res, y_res


# -------------------- SCALE --------------------
def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print("----------------Finished scaling data----------------")

    return X_train_scaled, X_test_scaled, scaler


# -------------------- PIPELINE --------------------
def build_pipeline(path):

    data = load_data(path)
    data = clean_data(data)

    X_train, X_test, y_train, y_test = split_data(data)

    X_train, X_test, y_train, dropped, selected_features = remove_correlated_features(
        X_train, X_test, y_train
    )

    X_train, y_train = balance_data(X_train, y_train)

    X_train, X_test, scaler = scale_data(X_train, X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "selected_features": selected_features,
        "dropped_features": dropped
    }