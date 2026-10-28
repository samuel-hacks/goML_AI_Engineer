"""Run the defence ablation on an eval set (sequentially) and write a markdown table.

    python eval/run_ablation.py --set eval/injection/test --out eval/injection_results.md [--only 1,3,5,6]

Predictions go to eval/runs/<set>_<config>.json, stderr logs to out/<set>_<config>.log.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from score_injection import HEADER, row, score  # noqa: E402

CONFIGS = [
    ("1", "no defences", ["--no-defences"]),
    ("2", "L3 prompt rules only", ["--no-hygiene", "--no-filter", "--no-spotlight", "--no-postcheck"]),
    ("3", "L0+L1 filter only", ["--no-spotlight", "--no-prompt-rules", "--no-postcheck"]),
    ("4", "L2 spotlighting only", ["--no-hygiene", "--no-filter", "--no-prompt-rules", "--no-postcheck"]),
    ("5", "L0+L1+L2+L3", ["--no-postcheck"]),
    ("6", "full stack (+L5)", []),
    ("6b", "full stack, 2nd run", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    name = a.set.name
    (ROOT / "out").mkdir(exist_ok=True)
    lines, results = [HEADER], {}
    for cid, label, flags in CONFIGS:
        if only and cid not in only:
            continue
        pred = HERE / "runs" / f"{name}_{cid}.json"
        logf = ROOT / "out" / f"{name}_{cid}.log"
        cmd = [sys.executable, str(ROOT / "extract.py"), "--inbox", str(a.set), "--output", str(pred), *flags]
        if a.model:
            cmd += ["--model", a.model]
        t0 = time.time()
        with logf.open("w") as lf:
            subprocess.run(cmd, stderr=lf, check=True)
        s = score(a.set, json.loads(pred.read_text()), logf.read_text())
        results[cid] = s
        lines.append(row(f"({cid}) {label}", s))
        print(lines[-1], f"  [{time.time() - t0:.0f}s]", flush=True)
    lines.append("| (7) full stack + classifier | not yet evaluated | | | | | | |")
    if "6" in results and "6b" in results:
        same = (HERE / "runs" / f"{name}_6.json").read_bytes() == (HERE / "runs" / f"{name}_6b.json").read_bytes()
        lines.append(f"\nFull stack run twice: outputs byte-identical = {same}")
    for cid, s in results.items():
        if s["asr"] or s["false_drop"] or s["clean_bad"]:
            lines.append(f"\n**({cid}) details**")
            lines += [f"- attack succeeded: {x}" for x in s["asr"]]
            lines += [f"- clean control wrong: {x}" for x in s["clean_bad"]]
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
