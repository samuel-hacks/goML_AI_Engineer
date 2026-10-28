"""Score a new list of registered students with the trained bundle. Never fits.

    python predict.py --input some_students.csv --output predictions.csv
        [--model PATH] [--force-model auto|clean|status]

Writes ``student_id,placed`` (one row per input row, input order, placed in {0,1}).
Diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the pickled placement_lib classes resolve
import placement_lib as pl  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent / "model" / "model.joblib"
EXPECTED = [pl.ID_COL, *pl.CAT_COLS, *pl.NUM_COLS, pl.STATUS_COL]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict placement for a CSV of students.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--force-model", choices=["auto", "clean", "status"], default="auto")
    args = ap.parse_args()

    if not args.model.is_file():
        eprint(f"error: model file not found: {args.model}")
        return 1
    try:
        df = pd.read_csv(args.input, dtype={pl.ID_COL: str}, skipinitialspace=True, keep_default_na=True)
    except Exception as exc:  # unreadable / missing / malformed input
        eprint(f"error: cannot read input {args.input}: {exc}")
        return 1
    bundle = joblib.load(args.model)

    df = pl.canonical_columns(df)
    missing = [c for c in EXPECTED if c not in df.columns]
    eprint(f"rows read: {len(df)}")
    if missing:
        eprint(f"missing columns filled with blanks: {', '.join(missing)}")
    ids = df[pl.ID_COL].astype("string").str.strip().fillna("") if pl.ID_COL in df else pd.Series([""] * len(df))

    pred, info = pl.score_frame(bundle, df.reset_index(drop=True), force=args.force_model)
    eprint(f"gate AUC: {info['gate_auc']:.3f} (threshold {bundle['gate']['threshold']}); "
           f"track: {info['track']}" + (" (forced)" if args.force_model != "auto" else ""))

    out = pd.DataFrame({pl.ID_COL: ids.to_numpy(dtype=object), "placed": pred.astype(int)})
    out.to_csv(args.output, index=False)
    eprint(f"wrote {len(out)} rows to {args.output} ({int(out.placed.sum())} predicted placed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
