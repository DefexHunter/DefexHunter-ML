import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




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


def remove_correlated_features(X, y, threshold=0.95, target_name="target"):
    X = X.copy()

    # 1. Correlation between features
    corr = X.corr(numeric_only=True)

    high_corr_pairs = []

    # 2. Find highly correlated feature pairs
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):

            correlation = corr.iloc[i, j]

            if abs(correlation) > threshold:
                high_corr_pairs.append(
                    (corr.columns[i], corr.columns[j], correlation)
                )

    print(f"Highly correlated pairs (|r| > {threshold}): {len(high_corr_pairs)}")

    for a, b, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
        print(f"{a:22s} <-> {b:22s} r = {r:.3f}")

    # 3. Correlation with target
    temp = X.copy()
    temp[target_name] = y

    target_corr = temp.corr(numeric_only=True)[target_name].abs()

    # 4. Decide which features to drop
    features_to_drop = set()

    for a, b, r in high_corr_pairs:

        # drop the one LESS correlated with target
        if target_corr[a] >= target_corr[b]:
            drop = b
        else:
            drop = a

        features_to_drop.add(drop)

    # 5. Build selected feature list
    selected_features = [col for col in X.columns if col not in features_to_drop]

    print(f"\nDropped ({len(features_to_drop)} features):")
    print(sorted(features_to_drop))

    print(f"\nRemaining features: {len(selected_features)}")

    X_clean = X[selected_features]

    print("----------------Finished removing correlated features----------------")

    return X_clean, y, list(features_to_drop)


def balance_data(
    X,
    y,
    max_majority_samples=3000,
    max_minority_samples=2103
):
    class_counts = y.value_counts()

    minority_n = min(int(class_counts.min()), max_minority_samples )
    majority_n = min(int(class_counts.max()), max_majority_samples )

    print(
        f"Resampling targets → "
        f"majority: {majority_n}, "
        f"minority: {minority_n}"
    )

    sampler = NearMiss(
        version=1,
        sampling_strategy={
            0: majority_n,
            1: minority_n
        }
    )

    X_balanced, y_balanced = sampler.fit_resample(X, y)

    print("\nClass distribution after NearMiss:")
    print(pd.Series(y_balanced).value_counts())

    print("----------------Finished balancing data----------------")

    return X_balanced, y_balanced

    

def split_and_scale(X, y, test_size=0.30, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Train: {len(y_train)} samples " f"| Test: {len(y_test)} samples")

    print("\nTrain class distribution:")
    print(pd.Series(y_train).value_counts())

    print("\nTest class distribution:")
    print(pd.Series(y_test).value_counts())

    print("----------------Finished split and scaling----------------")

    return (X_train, X_test, y_train, y_test,scaler)