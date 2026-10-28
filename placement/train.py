"""Train, compare and select the placement models; write model/model.joblib and reports/.

    python train.py                      # full run (~minutes)
    python train.py --out-dir /tmp/x --n-trials 5 --skip-gate-sim   # quick run used by the tests

Protocol (see analysis.ipynb for the reasoning):
* every candidate in both tracks is scored on the same 5x5 repeated stratified splits (seed 42);
* boosters are tuned with Optuna on a different split seed (7) against log-loss, never F1;
* blend weights / variants and the stacker are chosen on seed 7 as well;
* headline metric = nested F1 (threshold picked on the other four folds).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import placement_lib as pl

HERE = Path(__file__).resolve().parent
TRACKS = {"clean": False, "status": True}
BOOSTERS = ("LGBM", "XGB")
GATE_GRID = np.round(np.arange(0.55, 0.851, 0.01), 2)
warnings.filterwarnings("ignore", category=UserWarning)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- candidate search
def lr_grid() -> list[dict[str, Any]]:
    grid = []
    for c in (0.01, 0.03, 0.1, 0.3, 1, 3, 10):
        for l1 in (0.0, 0.2, 0.5, 0.8):  # 0.0 = pure L2, otherwise elastic-net
            for cw in (None, "balanced"):
                grid.append({"C": c, "l1_ratio": l1, "class_weight": cw})
    return grid


def select_linear(X: pd.DataFrame, y: np.ndarray, splits: pl.Splits, use_status: bool, n_jobs: int) -> tuple[dict, pd.DataFrame]:
    """Pick LogReg hyper-parameters by nested F1, then test city_tier as one-hot."""
    rows = []
    for params in lr_grid():
        oof = pl.get_oof(pl.make_linear(params, use_status), X, y, splits, n_jobs)
        nf = pl.nested_f1(oof, y, splits)
        rows.append({**params, "city_onehot": False, "nested_f1_mean": nf.mean(), "nested_f1_std": nf.std(ddof=1)})
    grid = pd.DataFrame(rows)
    best = grid.sort_values("nested_f1_mean", ascending=False, kind="stable").iloc[0]
    params = {"C": float(best.C), "l1_ratio": float(best.l1_ratio),
              "class_weight": None if pd.isna(best.class_weight) else best.class_weight}
    oof = pl.get_oof(pl.make_linear(params, use_status, city_onehot=True), X, y, splits, n_jobs)
    nf = pl.nested_f1(oof, y, splits)
    grid = pd.concat([grid, pd.DataFrame([{**params, "city_onehot": True, "nested_f1_mean": nf.mean(),
                                            "nested_f1_std": nf.std(ddof=1)}])], ignore_index=True)
    city_onehot = bool(nf.mean() > best.nested_f1_mean)
    return {"kind": "A", "params": params, "city_onehot": city_onehot}, grid


def suggest(trial: optuna.Trial, kind: str) -> dict[str, Any]:
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, pl.POS_WEIGHT]),
    }
    if kind == "LGBM":
        common |= {"num_leaves": trial.suggest_int("num_leaves", 3, 15),
                   "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                   "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    else:
        common |= {"min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
                   "gamma": trial.suggest_float("gamma", 0, 2),
                   "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
    return common


def tune_booster(kind: str, X: pd.DataFrame, y: np.ndarray, splits7: pl.Splits, use_status: bool,
                 n_trials: int, n_jobs: int) -> dict[str, Any]:
    """Optuna on the seed-7 split, minimising mean OOF log-loss."""
    def objective(trial: optuna.Trial) -> float:
        spec = {"kind": kind, "params": suggest(trial, kind)}
        oof = pl.get_oof(pl.build_model(spec, use_status), X, y, splits7, n_jobs)[0]
        return float(log_loss(y, oof))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=pl.SEED))
    study.optimize(objective, n_trials=n_trials)
    return {"kind": kind, "params": study.best_params}


def blend_variants(members: list[dict]) -> list[dict]:
    """Mean, rank-average and weighted means (w = weight on LogReg, rest split equally)."""
    k = len(members)
    out = [{"kind": "BLEND", "members": members, "method": "mean", "weights": [1.0] * k},
           {"kind": "BLEND", "members": members, "method": "rank", "weights": [1.0] * k}]
    for w in (0.3, 0.5, 0.7):
        out.append({"kind": "BLEND", "members": members, "method": "mean",
                    "weights": [w] + [(1 - w) / (k - 1)] * (k - 1)})
    return out


def choose_on_seed7(specs: list[dict], X: pd.DataFrame, y: np.ndarray, splits7: pl.Splits,
                    use_status: bool, n_jobs: int) -> dict:
    scores = [pl.nested_f1(pl.get_oof(pl.build_model(s, use_status), X, y, splits7, n_jobs), y, splits7).mean()
              for s in specs]
    return specs[int(np.argmax(scores))]


# --------------------------------------------------------------------------- per-track run
def run_track(track: str, X: pd.DataFrame, y: np.ndarray, splits: pl.Splits, splits7: pl.Splits,
              args: argparse.Namespace, reports: Path) -> dict[str, Any]:
    use_status = TRACKS[track]
    log(f"\n=== Track {track} ===")
    spec_a, grid = select_linear(X, y, splits, use_status, args.n_jobs)
    grid.assign(track=track).to_csv(reports / f"lr_grid_{track}.csv", index=False)
    log(f"A: {spec_a}")

    specs: dict[str, dict] = {"A": spec_a}
    for kind in BOOSTERS:
        specs[kind] = tune_booster(kind, X, y, splits7, use_status, args.n_trials, args.n_jobs)
        log(f"{kind} tuned: {specs[kind]['params']}")
    for name, members in {"C1": ["A", "LGBM"], "C2": ["A", "XGB"], "C3": ["A", "LGBM", "XGB"]}.items():
        specs[name] = choose_on_seed7(blend_variants([specs[m] for m in members]), X, y, splits7, use_status, args.n_jobs)
        log(f"{name}: method={specs[name]['method']} weights={np.round(specs[name]['weights'], 3).tolist()}")
    specs["ST"] = {"kind": "STACK", "members": [specs["A"], specs["LGBM"], specs["XGB"]]}

    oofs: dict[str, np.ndarray] = {}
    folds: dict[str, np.ndarray] = {}
    rows = []
    for name, spec in specs.items():
        oofs[name] = pl.get_oof(pl.build_model(spec, use_status), X, y, splits, args.n_jobs)
        folds[name] = pl.nested_f1(oofs[name], y, splits)
        np.save(reports / f"oof_{track}_{name}.npy", oofs[name])
        rows.append({"track": track, "candidate": name, **pl.summarize(oofs[name], y, splits),
                     **pl.paired_vs(folds[name], folds["A"]), "params": json.dumps(spec)})
        log(f"{name:5s} nested F1 {folds[name].mean():.4f} ± {folds[name].std(ddof=1):.4f}")
    res = pd.DataFrame(rows)

    # Selection rule: top candidate must beat A by >= 0.01 nested F1 and win >= 15/25 paired folds.
    top = res.sort_values("nested_f1_mean", ascending=False, kind="stable").iloc[0]
    winner = top.candidate if (top.candidate != "A" and top.mean_diff_vs_A >= 0.01 and top.wins_vs_A >= 15) else "A"
    res["selected"] = res.candidate == winner
    log(f"top by nested F1: {top.candidate}; shipped: {winner}")

    checks = []
    final_spec = dict(specs[winner])
    final_oof, final_nf = oofs[winner], folds[winner]
    if specs[winner]["kind"] != "A":  # calibration check for a non-linear winner
        cal_spec = {**final_spec, "calibrated": True}
        cal_oof = pl.get_oof(pl.build_model(cal_spec, use_status), X, y, splits, args.n_jobs)
        cal_nf = pl.nested_f1(cal_oof, y, splits)
        checks.append({"track": track, "check": "sigmoid calibration", "base": final_nf.mean(), "variant": cal_nf.mean()})
        if cal_nf.mean() > final_nf.mean():
            final_spec, final_oof, final_nf = cal_spec, cal_oof, cal_nf
    feat_spec = {**final_spec, "add_features": True}
    feat_oof = pl.get_oof(pl.build_model(feat_spec, use_status), X, y, splits, args.n_jobs)
    feat_nf = pl.nested_f1(feat_oof, y, splits)
    keep = feat_nf.mean() - final_nf.mean() >= 0.005
    checks.append({"track": track, "check": "engineered features", "base": final_nf.mean(), "variant": feat_nf.mean(),
                   "wins": int((feat_nf > final_nf).sum())})
    log(f"engineered features: {final_nf.mean():.4f} -> {feat_nf.mean():.4f}; kept={keep}")
    if keep:
        final_spec, final_oof, final_nf = feat_spec, feat_oof, feat_nf
    np.save(reports / f"oof_{track}_final.npy", final_oof)

    threshold, pooled_f1 = pl.best_threshold(y, final_oof.mean(axis=0))
    pipeline = pl.build_model(final_spec, use_status).fit(X, y)
    log(f"final {track}: {winner}, threshold {threshold:.3f} (pooled OOF F1 {pooled_f1:.4f})")
    fold_df = pd.DataFrame({name: f for name, f in folds.items()}).assign(track=track, fold=np.arange(len(splits)))
    return {"results": res, "checks": pd.DataFrame(checks), "folds": fold_df, "spec": final_spec, "name": winner,
            "pipeline": pipeline, "threshold": threshold, "nested_f1": float(final_nf.mean()),
            "nested_folds": final_nf, "oof": final_oof}


# --------------------------------------------------------------------------- gate simulation and robustness
def inner_threshold(spec: dict, use_status: bool, X: pd.DataFrame, y: np.ndarray) -> float:
    splits = list(StratifiedKFold(pl.N_SPLITS, shuffle=True, random_state=pl.SEED).split(X, y))
    oof = pl.get_oof(pl.build_model(spec, use_status), X, y, splits, n_jobs=1)[0]
    return pl.best_threshold(y, oof)[0]


def _sim_one(n: int, rep: int, tr: np.ndarray, te: np.ndarray, X: pd.DataFrame, y: np.ndarray,
             spec_c: dict, spec_s: dict) -> list[dict]:
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te].reset_index(drop=True), y[te]
    m_c = pl.build_model(spec_c, False).fit(Xtr, ytr)
    m_s = pl.build_model(spec_s, True).fit(Xtr, ytr)
    thr_c, thr_s = inner_threshold(spec_c, False, Xtr, ytr), inner_threshold(spec_s, True, Xtr, ytr)
    rng = np.random.default_rng(pl.SEED + 1000 * n + rep)
    scenarios = {
        "a_as_is": Xte,
        "b_shuffled": Xte.assign(status_code=rng.permutation(Xte.status_code.to_numpy())),
        "c_blank": Xte.assign(status_code=np.nan),
        "d_half_blank": Xte.assign(status_code=Xte.status_code.mask(rng.random(len(Xte)) < 0.5)),
    }
    rows = []
    for scen, df in scenarios.items():
        tp = pl.track_predictions(lambda d: m_c.predict_proba(d)[:, 1], lambda d: m_s.predict_proba(d)[:, 1],
                                  df, thr_c, thr_s)
        _, auc = pl.choose_track(tp["norm"], tp["p_clean"], gate_threshold=0.0)
        rows.append({"n": n, "rep": rep, "scenario": scen, "gate_auc": auc,
                     "clean_f1": f1_score(yte, tp["clean"]), "status_only_f1": f1_score(yte, tp["status_only"]),
                     "status_fallback_f1": f1_score(yte, tp["status_fallback"])})
    return rows


def gate_simulation(X: pd.DataFrame, y: np.ndarray, spec_c: dict, spec_s: dict, reps: int, n_jobs: int) -> pd.DataFrame:
    jobs = []
    for n in (100, 250, 500):
        sss = StratifiedShuffleSplit(n_splits=reps, test_size=n, random_state=pl.SEED)
        jobs += [(n, rep, tr, te) for rep, (tr, te) in enumerate(sss.split(X, y))]
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs)(delayed(_sim_one)(n, r, tr, te, X, y, spec_c, spec_s) for n, r, tr, te in jobs)
    return pd.DataFrame([row for rows in out for row in rows])


def apply_gate(runs: pd.DataFrame, thr: float) -> pd.DataFrame:
    status = runs.gate_auc.fillna(0) > thr
    return runs.assign(chose_status=status, gated_f1=np.where(status, runs.status_fallback_f1, runs.clean_f1))


def pick_gate_threshold(runs: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Max mean gated F1 subject to: status in >=95% of (a) runs at n>=250 and <=5% of (b)/(c) runs."""
    rows = []
    for thr in GATE_GRID:
        g = apply_gate(runs, thr)
        a = g[(g.scenario == "a_as_is") & (g.n >= 250)].chose_status.mean()
        bc = g[g.scenario.isin(["b_shuffled", "c_blank"])].chose_status.mean()
        rows.append({"gate_threshold": thr, "mean_gated_f1": g.gated_f1.mean(), "status_rate_a_n250plus": a,
                     "status_rate_bc": bc, "feasible": a >= 0.95 and bc <= 0.05})
    sweep = pd.DataFrame(rows)
    feas = sweep[sweep.feasible]
    best = feas[np.isclose(feas.mean_gated_f1, feas.mean_gated_f1.max())]
    # Ties: take the middle of the best plateau to keep margin on both sides.
    return float(best.gate_threshold.iloc[len(best) // 2]), sweep


def robustness(X: pd.DataFrame, y: np.ndarray, splits: pl.Splits, final: dict[str, dict], n_jobs: int) -> pd.DataFrame:
    """Blank a share of feature cells in each held-out fold; thresholds as in nested F1."""
    feat_cols = pl.NUM_COLS + pl.CAT_COLS + [pl.STATUS_COL]

    def one(i: int, tr: np.ndarray, te: np.ndarray) -> list[dict]:
        m_c = pl.build_model(final["clean"]["spec"], False).fit(X.iloc[tr], y[tr])
        m_s = pl.build_model(final["status"]["spec"], True).fit(X.iloc[tr], y[tr])
        thr = {}
        for t in TRACKS:
            other = np.ones(len(y), dtype=bool)
            other[te] = False
            thr[t] = pl.best_threshold(y[other], final[t]["oof"][i // pl.N_SPLITS][other])[0]
        rows = []
        for frac in (0.0, 0.1, 0.2):
            rng = np.random.default_rng(pl.SEED + i)
            Xte = X.iloc[te].reset_index(drop=True)
            mask = rng.random((len(Xte), len(feat_cols))) < frac
            Xte[feat_cols] = Xte[feat_cols].astype(object).mask(mask)
            tp = pl.track_predictions(lambda d: m_c.predict_proba(d)[:, 1], lambda d: m_s.predict_proba(d)[:, 1],
                                      Xte, thr["clean"], thr["status"])
            rows += [{"fold": i, "blank_frac": frac, "track": "clean", "f1": f1_score(y[te], tp["clean"])},
                     {"fold": i, "blank_frac": frac, "track": "status", "f1": f1_score(y[te], tp["status_fallback"])}]
        return rows

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs)(delayed(one)(i, tr, te) for i, (tr, te) in enumerate(splits))
    return pd.DataFrame([r for rows in out for r in rows])


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "train.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--sim-reps", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--skip-gate-sim", action="store_true", help="skip gate simulation/robustness (tests only)")
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    reports, model_dir = args.out_dir / "reports", args.out_dir / "model"
    reports.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, dtype={pl.ID_COL: str})
    y = df[pl.TARGET].to_numpy()
    X = df.drop(columns=[pl.TARGET])
    splits, splits7 = pl.make_splits(y), pl.tuning_splits(y)
    log(f"rows={len(df)} placed={y.sum()} ({y.mean():.3f}); {len(splits)} folds")

    final = {t: run_track(t, X, y, splits, splits7, args, reports) for t in TRACKS}
    pd.concat([final[t]["results"] for t in TRACKS]).to_csv(reports / "cv_results.csv", index=False)
    pd.concat([final[t]["checks"] for t in TRACKS]).to_csv(reports / "feature_checks.csv", index=False)
    pd.concat([final[t]["folds"] for t in TRACKS]).to_csv(reports / "fold_scores.csv", index=False)

    gate_thr = 0.68
    if not args.skip_gate_sim:
        log("\n=== Gate simulation ===")
        runs = gate_simulation(X, y, final["clean"]["spec"], final["status"]["spec"], args.sim_reps, args.n_jobs)
        runs.to_csv(reports / "gate_runs.csv", index=False)
        gate_thr, sweep = pick_gate_threshold(runs)
        sweep.to_csv(reports / "gate_sweep.csv", index=False)
        summary = (apply_gate(runs, gate_thr).groupby(["scenario", "n"])
                   .agg(gate_auc=("gate_auc", "mean"), status_rate=("chose_status", "mean"),
                        gated_f1=("gated_f1", "mean"), clean_f1=("clean_f1", "mean"),
                        status_only_f1=("status_only_f1", "mean")).reset_index().assign(gate_threshold=gate_thr))
        summary.to_csv(reports / "gate_sim.csv", index=False)
        log(f"gate threshold = {gate_thr}")
        log(summary.round(3).to_string(index=False))
        log("\n=== Robustness ===")
        rob = robustness(X, y, splits, final, args.n_jobs)
        rob.to_csv(reports / "robustness.csv", index=False)
        log(rob.groupby(["track", "blank_frac"]).f1.mean().round(4).to_string())

    import lightgbm, xgboost
    bundle = {
        t: {k: final[t][k] for k in ("pipeline", "threshold", "name", "nested_f1", "spec")} for t in TRACKS
    } | {
        "gate": {"threshold": gate_thr, "min_rows": 30},
        "meta": {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
                 "python": sys.version.split()[0], "seed": pl.SEED, "trained_on_rows": len(df)},
    }
    joblib.dump(bundle, model_dir / "model.joblib")
    log(f"\nsaved {model_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
