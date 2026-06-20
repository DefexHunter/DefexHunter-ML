import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from data.config import (
    TEST_SIZE,
    RANDOM_STATE,
    CORRELATION_THRESHOLD,
    TARGET_COLUMN,
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
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")
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
    y_train_reset = y_train.reset_index(drop=True)
    temp[target_name] = y_train_reset

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


# -------------------- PIPELINE --------------------
def build_pipeline(path):

    data = load_data(path)
    data = clean_data(data)

    X_train, X_test, y_train, y_test = split_data(data)

    X_train, X_test, y_train, dropped, selected_features = remove_correlated_features(
        X_train, X_test, y_train
    )

    selected_features = sorted(selected_features)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "selected_features": selected_features,
        "dropped_features": dropped,
    }