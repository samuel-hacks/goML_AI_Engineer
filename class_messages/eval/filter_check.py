"""Offline check of the deterministic L1 filter (no model): per message, which sentences
are dropped, whether the legit task sentence survives, and whether attack payload remains.

    python eval/filter_check.py eval/injection/dev [--quiet]
"""
import io
import json
import sys
import unicodedata
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extract as X  # noqa: E402


def n(s):
    return unicodedata.normalize("NFKC", X.hygiene(s)).lower()


def main(set_dir: Path, quiet: bool):
    labels = json.loads((set_dir / "labels.json").read_text(encoding="utf-8"))
    cfg = X.Config()
    miss, lost, clean_hit = [], [], []
    for path in sorted(set_dir.glob("*.txt"), key=lambda p: X.natural_key(p.name)):
        msg = X.parse_message(path)
        lab = labels[msg.message_id]
        with redirect_stderr(io.StringIO()):
            f = X.filter_text(X.hygiene(msg.body), msg.message_id, cfg)
            hdr = {h: X.prepare_header(msg.headers.get(h, ""), msg.message_id, h, cfg) for h in ("from", "date")}
        kept = n(f.text + " " + hdr["from"] + " " + hdr["date"])
        left = [p for p in lab["payload"] if n(p) in kept]
        attack_removed = bool(f.removed) or "[removed]" in hdr.values()
        if lab["kind"] == "attack" and (left or not attack_removed):
            miss.append(f"{msg.message_id} [{lab['category']}] payload left={left} removed={f.removed}")
        if lab["key"] and n(lab["key"]) not in n(f.text):
            lost.append(f"{msg.message_id} [{lab['category']}] legit task lost; removed={f.removed}")
        if lab["kind"] == "clean" and f.removed:
            clean_hit.append(f"{msg.message_id}: removed={f.removed}")
        if not quiet:
            print(f"{msg.message_id}: kept={f.text!r} hdr={hdr}")
    n_att = sum(l["kind"] == "attack" for l in labels.values())
    n_clean = sum(l["kind"] == "clean" for l in labels.values())
    print(f"\nattacks not fully removed by L1: {len(miss)}/{n_att}", *miss, sep="\n  ")
    print(f"legit task sentence lost: {len(lost)}", *lost, sep="\n  ")
    print(f"clean controls with a sentence removed: {len(clean_hit)}/{n_clean}", *clean_hit, sep="\n  ")


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--quiet" in sys.argv)
