import numpy as np
import pandas as pd

def load_data(path):
    dataset = pd.read_csv(path)
    print("Shape:", dataset.shape)
    print ("----------------Finished loading data----------------")
    return dataset


def clean_data(dataset):
    dataset['defects'] = dataset['defects'].astype('float')
    dataset.dropna(axis=0, inplace=True)
    print("Shape after cleaning:", dataset.shape)
    print ("----------------Finished cleaning data----------------")
    return dataset

def remove_correlated_features(dataset, threshold=0.95):
    dataset = dataset.copy()

    corr = dataset.corr(numeric_only=True)

    high_corr_pairs = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):

            correlation = corr.iloc[i, j]

            if abs(correlation) > threshold:

                high_corr_pairs.append(
                    (
                        corr.columns[i],
                        corr.columns[j],
                        round(correlation, 3)
                    )
                )

    print(f"Highly correlated pairs (|r| > {threshold}): {len(high_corr_pairs)}")

    for a, b, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
        print(f"{a:22s} <-> {b:22s} r = {r}")

    corr_with_target = corr["defects"].abs()

    features_to_drop = set()

    for a, b, r in high_corr_pairs:

        drop = (
            b
            if corr_with_target.get(a, 0) >= corr_with_target.get(b, 0)
            else a
        )

        if drop != "defects":
            features_to_drop.add(drop)

    feature_cols_final = [
        col
        for col in dataset.columns
        if col not in features_to_drop and col != "defects"
    ]

    print(f"\nDropped ({len(features_to_drop)}):")
    print(sorted(features_to_drop))

    print(f"\nKept ({len(feature_cols_final)} features)")

    X_clean = dataset[feature_cols_final]

    y_clean = dataset["defects"]

    print ("----------------Finished removing correlated features----------------")

    return X_clean, y_clean, list(features_to_drop)

