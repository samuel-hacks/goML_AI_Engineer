"""Shared code for the placement classifier.

Everything that is pickled into ``model/model.joblib`` (the Normalizer, the
blend wrapper) lives here so that ``train.py``, ``predict.py`` and the
notebook all import the same class by the same module path.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TUNE_SEED = 7
TARGET = "placed"
ID_COL = "student_id"
NUM_COLS = ["city_tier", "cgpa", "backlogs", "internship_months", "coding_score", "projects", "workshops"]
CAT_COLS = ["department"]
STATUS_COL = "status_code"
KNOWN_DEPTS = ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"]
CLIP: dict[str, tuple[float | None, float | None]] = {
    "cgpa": (0, 10), "coding_score": (0, 100), "backlogs": (0, None), "internship_months": (0, None),
    "projects": (0, None), "workshops": (0, None), "city_tier": (1, 3), "status_code": (0, 1),
}
ENGINEERED = ["cgpa_x_coding", "has_internship", "has_backlog"]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.80 + 1e-9, 0.005), 3)
N_SPLITS, N_REPEATS = 5, 5
# Positive-class weight that balances the training prior: 1798 negatives / 702 positives.
POS_WEIGHT = 2.56


# --------------------------------------------------------------------------- preprocessing
def canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and lower-case column names; on a name clash keep the first column."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out.loc[:, ~out.columns.duplicated()]


def numeric_columns(use_status: bool) -> list[str]:
    return NUM_COLS + ([STATUS_COL] if use_status else [])


class Normalizer(BaseEstimator, TransformerMixin):
    """Stateless cleaning step that runs first in every pipeline.

    Output columns are fixed regardless of the input: the numeric features
    (coerced and clipped), ``department`` (upper-cased, blank -> "MISSING"),
    one ``<col>_missing`` indicator per numeric feature, and optionally the
    engineered features. Unknown input columns (``placed``, ``student_id``,
    anything extra) are dropped; absent ones are created as NaN.
    """

    def __init__(self, use_status: bool = False, add_features: bool = False) -> None:
        self.use_status = use_status
        self.add_features = add_features

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Normalizer":
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        num = numeric_columns(self.use_status)
        extra = ENGINEERED if self.add_features else []
        return np.array(num + CAT_COLS + [f"{c}_missing" for c in num] + extra, dtype=object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = canonical_columns(pd.DataFrame(X))
        out = pd.DataFrame(index=df.index)
        num = numeric_columns(self.use_status)
        for col in num:
            raw = df[col] if col in df else pd.Series(np.nan, index=df.index)
            if not pd.api.types.is_numeric_dtype(raw):
                raw = raw.astype("string").str.strip()
            vals = pd.to_numeric(raw, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            lo, hi = CLIP[col]
            out[col] = vals.clip(lower=lo, upper=hi)
        dept = df["department"] if "department" in df else pd.Series(np.nan, index=df.index)
        dept = dept.astype("string").str.strip().str.upper()
        out["department"] = dept.mask(dept.isna() | (dept == ""), "MISSING").astype(object)
        for col in num:
            out[f"{col}_missing"] = out[col].isna().astype(float)
        if self.add_features:
            # NaN propagates: trees route it natively, the linear imputer fills it.
            out["cgpa_x_coding"] = out["cgpa"] * out["coding_score"]
            out["has_internship"] = (out["internship_months"] > 0).astype(float).where(out["internship_months"].notna())
            out["has_backlog"] = (out["backlogs"] > 0).astype(float).where(out["backlogs"].notna())
        return out


def feature_lists(use_status: bool, add_features: bool, city_onehot: bool = False) -> tuple[list[str], list[str], list[str]]:
    """(scaled numeric, missing indicators, one-hot categoricals) for the column transformer."""
    num = numeric_columns(use_status)
    ind = [f"{c}_missing" for c in num]
    numeric = num + (ENGINEERED if add_features else [])
    cat = list(CAT_COLS)
    if city_onehot:
        numeric = [c for c in numeric if c != "city_tier"]
        cat.append("city_tier")
    return numeric, ind, cat


# --------------------------------------------------------------------------- model builders
def make_linear(params: dict[str, Any], use_status: bool, add_features: bool = False, city_onehot: bool = False) -> Pipeline:
    """Normalizer -> median impute + scale | indicators | one-hot -> LogisticRegression.

    ``params`` holds C, l1_ratio (0 = pure L2) and class_weight.
    """
    numeric, ind, cat = feature_lists(use_status, add_features, city_onehot)
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("scale", StandardScaler())]), numeric),
        ("ind", "passthrough", ind),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    l1_ratio = float(params.get("l1_ratio", 0.0))
    clf = LogisticRegression(
        C=params["C"], l1_ratio=l1_ratio, class_weight=params.get("class_weight"),
        solver="saga" if l1_ratio > 0 else "lbfgs", max_iter=5000, random_state=SEED,
    )
    return Pipeline([("norm", Normalizer(use_status, add_features)), ("pre", pre), ("clf", clf)])


def _tree_pipeline(model: BaseEstimator, use_status: bool, add_features: bool) -> Pipeline:
    numeric, ind, cat = feature_lists(use_status, add_features)
    pre = ColumnTransformer([
        ("num", "passthrough", numeric + ind),  # NaN left in place for the booster
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    return Pipeline([("norm", Normalizer(use_status, add_features)), ("pre", pre), ("clf", model)])


def make_lgbm(params: dict[str, Any], use_status: bool, add_features: bool = False) -> Pipeline:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(objective="binary", random_state=SEED, deterministic=True, force_row_wise=True,
                           n_jobs=4, verbose=-1, subsample_freq=1, **params)
    return _tree_pipeline(model, use_status, add_features)


def make_xgb(params: dict[str, Any], use_status: bool, add_features: bool = False) -> Pipeline:
    from xgboost import XGBClassifier

    model = XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=4, eval_metric="logloss", **params)
    return _tree_pipeline(model, use_status, add_features)


class BlendClassifier(ClassifierMixin, BaseEstimator):
    """Weighted average of base-model scores.

    method="mean" averages probabilities. method="rank" first maps each
    model's probability to its percentile among that model's predictions on
    the training rows, so the transform is fixed at fit time and does not
    depend on the batch being scored. Rank scores are not probabilities.
    """

    def __init__(self, estimators: Sequence[BaseEstimator], weights: Sequence[float], method: str = "mean") -> None:
        self.estimators = estimators
        self.weights = weights
        self.method = method

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BlendClassifier":
        self.estimators_ = [clone(e).fit(X, y) for e in self.estimators]
        self.classes_ = np.array([0, 1])
        if self.method == "rank":
            self.reference_ = [np.sort(e.predict_proba(X)[:, 1]) for e in self.estimators_]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = []
        for i, est in enumerate(self.estimators_):
            p = est.predict_proba(X)[:, 1]
            if self.method == "rank":
                ref = self.reference_[i]
                p = np.searchsorted(ref, p, side="right") / len(ref)
            scores.append(p)
        w = np.asarray(self.weights, dtype=float)
        s = np.clip(np.average(np.vstack(scores), axis=0, weights=w), 0.0, 1.0)
        return np.column_stack([1 - s, s])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model(spec: dict[str, Any], use_status: bool) -> BaseEstimator:
    """Rebuild an unfitted estimator from a JSON-serialisable candidate spec."""
    kind = spec["kind"]
    feats = bool(spec.get("add_features", False))
    if kind == "A":
        model: BaseEstimator = make_linear(spec["params"], use_status, feats, bool(spec.get("city_onehot", False)))
    elif kind == "LGBM":
        model = make_lgbm(spec["params"], use_status, feats)
    elif kind == "XGB":
        model = make_xgb(spec["params"], use_status, feats)
    elif kind == "BLEND":
        members = [build_model({**m, "add_features": feats}, use_status) for m in spec["members"]]
        model = BlendClassifier(members, spec["weights"], spec["method"])
    elif kind == "STACK":
        members = [(m["kind"].lower(), build_model({**m, "add_features": feats}, use_status)) for m in spec["members"]]
        model = StackingClassifier(
            members, final_estimator=LogisticRegression(C=1.0, max_iter=5000),
            cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=TUNE_SEED), stack_method="predict_proba",
        )
    else:
        raise ValueError(f"unknown candidate kind {kind!r}")
    if spec.get("calibrated"):
        model = CalibratedClassifierCV(model, method="sigmoid", cv=5)
    return model


# --------------------------------------------------------------------------- validation harness
Splits = list[tuple[np.ndarray, np.ndarray]]


def make_splits(y: np.ndarray, seed: int = SEED, n_repeats: int = N_REPEATS) -> Splits:
    """Repeat-major list of (train_idx, test_idx); fold f of repeat r is at r * N_SPLITS + f."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def tuning_splits(y: np.ndarray) -> Splits:
    """Single 5-fold split on a different seed, used only for tuning and blend/stack choices."""
    return list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=TUNE_SEED).split(np.zeros(len(y)), y))


def _fit_predict(est: BaseEstimator, X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    model = clone(est).fit(X.iloc[tr], y[tr])
    return model.predict_proba(X.iloc[te])[:, 1]


def get_oof(est: BaseEstimator, X: pd.DataFrame, y: np.ndarray, splits: Splits, n_jobs: int = 3) -> np.ndarray:
    """Out-of-fold probabilities, shape (n_repeats, n_samples)."""
    preds = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(est, X, y, tr, te) for tr, te in splits)
    n_rep = len(splits) // N_SPLITS
    oof = np.full((n_rep, len(y)), np.nan)
    for i, ((_, te), p) in enumerate(zip(splits, preds)):
        oof[i // N_SPLITS, te] = p
    return oof


def f1_curve(y: np.ndarray, p: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> np.ndarray:
    """F1 for placed=1 at every threshold of the grid (vectorised, predict 1 when p >= t)."""
    pred = p[None, :] >= grid[:, None]
    tp = (pred & (y[None, :] == 1)).sum(axis=1)
    fp = (pred & (y[None, :] == 0)).sum(axis=1)
    fn = (~pred & (y[None, :] == 1)).sum(axis=1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(threshold, F1) maximising F1 on the grid; ties go to the lowest threshold."""
    curve = f1_curve(y, p)
    i = int(np.argmax(curve))
    return float(THRESHOLD_GRID[i]), float(curve[i])


def nested_f1(oof: np.ndarray, y: np.ndarray, splits: Splits) -> np.ndarray:
    """Per-fold F1 where each fold's threshold is chosen on the other folds' OOF predictions."""
    scores = []
    for i, (_, te) in enumerate(splits):
        p = oof[i // N_SPLITS]
        other = np.ones(len(y), dtype=bool)
        other[te] = False
        thr, _ = best_threshold(y[other], p[other])
        scores.append(f1_score(y[te], (p[te] >= thr).astype(int), zero_division=0))
    return np.array(scores)


def summarize(oof: np.ndarray, y: np.ndarray, splits: Splits) -> dict[str, float]:
    """Headline metrics: nested F1 (mean/std over folds), per-repeat means of the rest, pooled threshold."""
    nf = nested_f1(oof, y, splits)
    per_rep = [(roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p), *best_threshold(y, p))
               for p in oof]
    auc, pr_auc, brier, _, best_f1 = np.mean(per_rep, axis=0)
    thr, _ = best_threshold(y, oof.mean(axis=0))
    return {"nested_f1_mean": float(nf.mean()), "nested_f1_std": float(nf.std(ddof=1)), "auc": float(auc),
            "pr_auc": float(pr_auc), "brier": float(brier), "best_f1": float(best_f1), "best_threshold": thr}


def paired_vs(nf_cand: np.ndarray, nf_ref: np.ndarray) -> dict[str, float]:
    diff = nf_cand - nf_ref
    return {"mean_diff_vs_A": float(diff.mean()), "std_diff_vs_A": float(diff.std(ddof=1)),
            "wins_vs_A": int((diff > 0).sum())}


# --------------------------------------------------------------------------- validity gate and scoring
def choose_track(df_norm: pd.DataFrame, p_clean: np.ndarray, gate_threshold: float, min_rows: int = 30) -> tuple[str, float]:
    """Use status_code only if it ranks like it did in training (AUC of clean probabilities vs status_code)."""
    s = df_norm[STATUS_COL]
    m = s.notna().to_numpy()
    if m.sum() < min_rows or s[m].nunique() < 2:
        return "clean", float("nan")
    auc = roc_auc_score(s[m].astype(int), p_clean[m])
    return ("status" if auc > gate_threshold else "clean"), float(auc)


def score_frame(bundle: dict[str, Any], df: pd.DataFrame, force: str = "auto") -> tuple[np.ndarray, dict[str, Any]]:
    """0/1 predictions for every row plus a small info dict (track, gate AUC). Never fits anything."""
    info: dict[str, Any] = {"track": "clean", "gate_auc": float("nan")}
    if len(df) == 0:
        return np.zeros(0, dtype=int), info
    clean, status = bundle["clean"], bundle["status"]
    p_clean = clean["pipeline"].predict_proba(df)[:, 1]
    pred = (p_clean >= clean["threshold"]).astype(int)
    norm = Normalizer(use_status=True).transform(df)
    track, auc = choose_track(norm, p_clean, bundle["gate"]["threshold"], bundle["gate"]["min_rows"])
    info["gate_auc"] = auc
    track = track if force == "auto" else force
    info["track"] = track
    if track == "status":
        has = norm[STATUS_COL].notna().to_numpy()  # rows without status_code keep the clean prediction
        if has.any():
            p_status = status["pipeline"].predict_proba(df.iloc[np.flatnonzero(has)])[:, 1]
            pred[has] = (p_status >= status["threshold"]).astype(int)
    return pred, info


def track_predictions(
    predict_clean: Callable[[pd.DataFrame], np.ndarray], predict_status: Callable[[pd.DataFrame], np.ndarray],
    df: pd.DataFrame, thr_clean: float, thr_status: float,
) -> dict[str, Any]:
    """Clean-only, status-only and status-with-fallback predictions for one test frame (used by the gate simulation)."""
    p_clean = predict_clean(df)
    p_status = predict_status(df)
    norm = Normalizer(use_status=True).transform(df)
    has = norm[STATUS_COL].notna().to_numpy()
    clean_pred = (p_clean >= thr_clean).astype(int)
    status_pred = (p_status >= thr_status).astype(int)
    return {"p_clean": p_clean, "norm": norm, "clean": clean_pred, "status_only": status_pred,
            "status_fallback": np.where(has, status_pred, clean_pred)}
