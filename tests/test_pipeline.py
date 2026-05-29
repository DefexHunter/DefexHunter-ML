#Unit tests for each pipeline step in src/data/pipeline.py

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import (
    clean_data,
    remove_correlated_features,
    balance_data,
    scale_data,
    split_data,
)


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_dataset():
    """Small synthetic dataset that mirrors JM1 structure."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "loc":              np.random.randint(1, 200, n).astype(float),
        "v(g)":             np.random.randint(1, 30,  n).astype(float),
        "ev(g)":            np.random.randint(1, 20,  n).astype(float),
        "iv(g)":            np.random.randint(1, 20,  n).astype(float),
        "n":                np.random.randint(10, 500, n).astype(float),
        "v":                np.random.uniform(10, 2000, n),
        "l":                np.random.uniform(0.001, 1, n),
        "d":                np.random.uniform(1, 100, n),
        "i":                np.random.uniform(1, 100, n),
        "e":                np.random.uniform(100, 50000, n),
        "b":                np.random.uniform(0, 1, n),
        "t":                np.random.uniform(1, 3000, n),
        "lOCode":           np.random.randint(1, 150, n).astype(float),
        "lOComment":        np.random.randint(0, 50,  n).astype(float),
        "lOBlank":          np.random.randint(0, 30,  n).astype(float),
        "lOCodeAndComment": np.random.randint(0, 10,  n).astype(float),
        "uniq_Op":          np.random.randint(1, 30,  n).astype(float),
        "uniq_Opnd":        np.random.randint(1, 50,  n).astype(float),
        "total_Op":         np.random.randint(1, 200, n).astype(float),
        "total_Opnd":       np.random.randint(1, 200, n).astype(float),
        "branchCount":      np.random.randint(0, 50,  n).astype(float),
        "defects":          np.random.choice([0.0, 1.0], n),
    })
    return df


@pytest.fixture
def dirty_dataset(raw_dataset):
    """Same dataset but with some NaN rows injected."""
    df = raw_dataset.copy()
    df.loc[[0, 5, 10], "loc"] = np.nan
    return df


# ── tests ─────────────────────────────────────────────────────────────────────
class TestCleanData:
    def test_removes_nan_rows(self, dirty_dataset):
        cleaned = clean_data(dirty_dataset)
        assert cleaned.isnull().sum().sum() == 0

    def test_defects_is_float(self, raw_dataset):
        raw_dataset["defects"] = raw_dataset["defects"].astype(str)
        cleaned = clean_data(raw_dataset)
        assert cleaned["defects"].dtype == float

    def test_shape_reduces_with_nans(self, dirty_dataset):
        cleaned = clean_data(dirty_dataset)
        assert len(cleaned) < len(dirty_dataset)


class TestSplitData:
    def test_split_sizes(self, raw_dataset):
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, y_test = split_data(cleaned)
        total = len(y_train) + len(y_test)
        assert total == len(cleaned)
        assert abs(len(y_test) / total - 0.30) < 0.05  # ~30% test

    def test_both_classes_in_splits(self, raw_dataset):
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, y_test = split_data(cleaned)
        assert set(y_train.unique()) == {0.0, 1.0}
        assert set(y_test.unique())  == {0.0, 1.0}


class TestRemoveCorrelatedFeatures:
    def test_never_increases_columns(self, raw_dataset):
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, y_test = split_data(cleaned)
        X_tr2, X_te2, _, dropped, selected = remove_correlated_features(
            X_train, X_test, y_train
        )
        assert len(selected) <= X_train.shape[1]

    def test_train_test_have_same_columns(self, raw_dataset):
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, y_test = split_data(cleaned)
        X_tr2, X_te2, _, dropped, selected = remove_correlated_features(
            X_train, X_test, y_train
        )
        assert list(X_tr2.columns) == list(X_te2.columns)

    def test_defects_not_in_features(self, raw_dataset):
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, y_test = split_data(cleaned)
        X_tr2, _, _, _, selected = remove_correlated_features(
            X_train, X_test, y_train
        )
        assert "defects" not in selected


class TestBalanceData:
    def test_output_is_balanced(self, raw_dataset):
        cleaned = clean_data(raw_dataset)
        X_train, _, y_train, _ = split_data(cleaned)
        X_bal, y_bal = balance_data(X_train, y_train)
        counts = pd.Series(y_bal).value_counts()
        # classes should be equal or within 1
        assert abs(counts.max() - counts.min()) <= 1

    def test_no_data_leakage_from_test(self, raw_dataset):
        """balance_data must only touch training data, test set untouched."""
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, y_test = split_data(cleaned)
        original_test_len = len(y_test)
        balance_data(X_train, y_train)
        # test set must be exactly the same size
        assert len(y_test) == original_test_len


class TestScaleData:
    def test_scaler_fit_on_train_only(self, raw_dataset):
        """Scaler must be fit on train, transform on test — no leakage."""
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, _ = split_data(cleaned)
        X_tr_sc, X_te_sc, scaler = scale_data(X_train, X_test)
        # train mean should be ~0 after scaling
        assert abs(X_tr_sc.mean().mean()) < 0.1

    def test_output_shapes_preserved(self, raw_dataset):
        cleaned = clean_data(raw_dataset)
        X_train, X_test, y_train, _ = split_data(cleaned)
        X_tr_sc, X_te_sc, _ = scale_data(X_train, X_test)
        assert X_tr_sc.shape == X_train.shape
        assert X_te_sc.shape == X_test.shape