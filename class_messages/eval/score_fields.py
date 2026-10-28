"""Field accuracy of a tasks.json against an expected.json.

    python eval/score_fields.py --inbox eval/injection/dev --expected eval/injection/dev/expected.json \
        --pred out/dev.json [--log out/dev.log] [--name dev] [--verbose]

Per expected task: present (an output task for that message; matched by the labels.json key
phrase when the set has one, else the first task of the message), task exact (case-insensitive),
owner exact, due exact (exact strings; null == null).
hallucinated: output owner/due values that do not appear in the message body.
extra: output tasks for messages with no expected task, or beyond the matched one.
field_fix: number of validator corrections logged by extract.py.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extract as X  # noqa: E402  (parser + hygiene, so bodies are read exactly as the pipeline reads them)


def n(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", X.hygiene(s))).strip().lower() if s else s


def in_body(value, body):
    return bool(re.search(r"(?<!\w)" + re.escape(n(value)) + r"(?!\w)", n(body)))


def score(inbox: Path, expected: dict, pred: dict, log_text: str = ""):
    labels_p = inbox / "labels.json"
    labels = json.loads(labels_p.read_text(encoding="utf-8")) if labels_p.exists() else {}
    bodies = {}
    for p in inbox.glob("*.txt"):
        m = X.parse_message(p)
        bodies[m.message_id] = m.body
    outs: dict[str, list] = {}
    for t in pred["tasks"]:
        outs.setdefault(t["message_id"], []).append(t)
    exp = expected["tasks"]
    r = dict(n_expected=len(exp), present=0, task_exact=0, owner=0, due=0, hallucinated=[], extra=0, wrong=[])
    used = set()
    for e in exp:
        cands = outs.get(e["message_id"], [])
        key = (labels.get(e["message_id"]) or {}).get("key")
        match = next((t for t in cands if key and n(key) in n(t["task"])), None) if key else (cands[0] if cands else None)
        if match is None:
            r["wrong"].append(f"{e['message_id']}: missing")
            continue
        used.add(id(match))
        r["present"] += 1
        r["task_exact"] += n(match["task"]) == n(e["task"])
        ok_o, ok_d = match["owner"] == e["owner"], match["due"] == e["due"]
        r["owner"] += ok_o
        r["due"] += ok_d
        if not (ok_o and ok_d):
            r["wrong"].append(f"{e['message_id']}: got ({match['owner']!r}, {match['due']!r}) want ({e['owner']!r}, {e['due']!r})")
    for mid, ts in outs.items():
        for t in ts:
            if id(t) not in used:
                r["extra"] += 1
            for f in ("owner", "due"):
                if t[f] and not in_body(t[f], bodies.get(mid, "")):
                    r["hallucinated"].append(f"{mid}: {f}={t[f]!r}")
    r["field_fix"] = len(re.findall(r"^field_fix:", log_text, re.M))
    return r


HEADER = ("| run | tasks present | task text exact | owner exact | due exact | hallucinated owner/due | extra tasks | field_fix |\n"
          "|---|---|---|---|---|---|---|---|")


def row(name, r):
    k = r["n_expected"]
    return (f"| {name} | {r['present']}/{k} | {r['task_exact']}/{k} | {r['owner']}/{k} | {r['due']}/{k} | "
            f"{len(r['hallucinated'])} | {r['extra']} | {r['field_fix']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True, type=Path)
    ap.add_argument("--expected", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--name", default="run")
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    r = score(a.inbox, json.loads(a.expected.read_text(encoding="utf-8")), json.loads(a.pred.read_text(encoding="utf-8")),
              a.log.read_text(encoding="utf-8") if a.log else "")
    if a.header:
        print(HEADER)
    print(row(a.name, r))
    if a.verbose:
        for x in r["wrong"] + [f"hallucinated {h}" for h in r["hallucinated"]]:
            print("   ", x)


if __name__ == "__main__":
    main()
