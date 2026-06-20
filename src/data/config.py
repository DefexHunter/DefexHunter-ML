# src/data/config.py

# ── split ────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.30
RANDOM_STATE = 42

# ── feature selection ────────────────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.95
TARGET_COLUMN = "defects"

# ── class balancing ──────────────────────────────────────────────────────────
MAX_MAJORITY_SAMPLES = None
MAX_MINORITY_SAMPLES = None