"""Calibration -- the part that decides whether the probabilities are usable.

Measure (reliability table, ECE, Brier, log loss), fix (Platt or isotonic),
and always fit the calibrator on a HELD-OUT set, never on training or test.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-12


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


# ---- metrics ---------------------------------------------------------------

def brier_score(p, y) -> float:
    p, y = _clip(p), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss_score(p, y) -> float:
    p, y = _clip(p), np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    p, y = _clip(p), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf, acc = p[mask].mean(), y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_table(p, y, n_bins: int = 10):
    """Per-bin (mean predicted prob, observed win rate, count)."""
    p, y = _clip(p), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append((f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                     float(p[mask].mean()),
                     float(y[mask].mean()),
                     int(mask.sum())))
    return rows


# ---- calibrators -----------------------------------------------------------

def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic calibration on the model's log-odds. Robust on small samples."""

    def fit(self, p, y):
        z = _logit(p).reshape(-1, 1)
        self.lr = LogisticRegression().fit(z, np.asarray(y))
        return self

    def transform(self, p):
        z = _logit(p).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]


class IsotonicCalibrator:
    """Non-decreasing free-form map. More flexible; needs more data."""

    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.iso.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, p):
        return self.iso.predict(np.asarray(p, dtype=float))


# ---- optional plot ---------------------------------------------------------

def save_reliability_plot(p, y, path: str, n_bins: int = 10, title: str = ""):
    """Save a reliability diagram if matplotlib is available; else no-op."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = reliability_table(p, y, n_bins)
    conf = [r[1] for r in rows]
    acc = [r[2] for r in rows]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    plt.plot(conf, acc, "o-", label="model")
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed win rate")
    plt.title(title or "Reliability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return True
