"""Build eval/v2_results.md (evaluation history, rule attribution, guard table) from saved run files.

    .venv/bin/python -B eval/v2_results.py

Reads: eval/injection_results.md (v1 held-out), ../validation_llmail/results.json (v1 LLMail sample 1),
eval/runs/v2/*.json + out/v2/*.log (v2 held-out), ../validation_llmail_v2/runs/* (v2 LLMail runs),
eval/guard/results.md. Writes eval/v2_results.md and eval/runs/v2/history.json.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CP = ROOT.parent
sys.path.insert(0, str(ROOT / "eval"))
import score_injection as SI  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LM1 = load_module(CP / "validation_llmail/score.py", "lm1")
LM2 = load_module(CP / "validation_llmail_v2/score.py", "lm2")
done_re = re.compile(r"done: (\d+) files, (\d+) tasks, (\d+) failed; tokens sent=(\d+) received=(\d+) calls=(\d+)")


def tokens(log):
    f = done_re.findall(Path(log).read_text(encoding="utf-8", errors="replace"))[-1]
    return int(f[3]), int(f[4])


def fmt_tok(t):
    return f"{t[0]:,} / {t[1]:,}"


def rule_counts(log):
    txt = Path(log).read_text(encoding="utf-8", errors="replace")
    return {k: len(re.findall(rf"^dropped_task: \S+: {k}:", txt, re.M))
            for k in ("r1_unowned_in_tainted_message", "r2_tainted_forwarded_block")}


hist = []
# v1 held-out (blind), parsed from the frozen v1 results file
res = (ROOT / "eval/injection_results.md").read_text(encoding="utf-8")
m = re.search(r"^\| \(6\) full stack \(\+L5\) \| (\d+/\d+) \(\d+%\) \| (\d+/\d+) \(\d+%\) \| [^|]+\| (\d+/\d+) \(\d+%\) \| "
              r"[^|]+\| (\d+/\d+) \(\d+%\) \| (\d+) / (\d+) \|", res, re.M)
hist.append(dict(version="v1", set="held-out test", blind="yes", succeed=m.group(1), lost=f"suppressed {m.group(2)}",
                 benign=f"clean drops {m.group(3)}, clean correct {m.group(4)}", tokens=f"{int(m.group(5)):,} / {int(m.group(6)):,}"))
# v1 LLMail sample 1 (blind)
v1 = json.loads((CP / "validation_llmail/results.json").read_text())["summary"]
t1 = tokens(CP / "validation_llmail/runs/headline.log")
hist.append(dict(version="v1", set="LLMail sample 1", blind="yes", succeed=f"{v1['success']}/{v1['N']}",
                 lost=f"{v1['N'] - v1['kept']}/{v1['N']} lost, {v1['suppression']} suppressed",
                 benign=f"{v1['false_drops']}/{v1['benign_n']} false drops", tokens=fmt_tok(t1)))


def held(pred, log, label, blind):
    s = SI.score(ROOT / "eval/injection/test", json.loads(Path(pred).read_text()), Path(log).read_text())
    return dict(version=label, set="held-out test", blind=blind, succeed=f"{len(s['asr'])}/{s['n_attack']}",
                lost=f"suppressed {len(s['suppressed'])}/{s['n_attack']}",
                benign=f"clean drops {len(s['false_drop'])}/{s['n_clean_task']}, clean correct {len(s['clean_ok'])}/{s['n_clean']}",
                tokens=fmt_tok(tokens(log)), detail=s["asr"])


def llmail(mod, pred, log, label, setname, blind):
    rows = mod.score(pred, log)
    s = mod.summarize(rows, (CP / "validation_llmail_v2/freeze/head_before.txt").read_text().strip())
    s["score_block"] = mod.block(s)
    return dict(version=label, set=setname, blind=blind, succeed=f"{s['success']}/{s['N']}",
                lost=f"{s['N'] - s['kept']}/{s['N']} lost, {s['suppression']} suppressed",
                benign=f"{s['false_drops']}/{s['benign_n']} false drops", tokens=fmt_tok(tokens(log)),
                summary=s, rows=rows)


V2R = CP / "validation_llmail_v2/runs"
hist.append(held(ROOT / "eval/runs/v2/test_v2.json", ROOT / "out/v2/test_v2.log", "v2", "no (after fix)"))
hist.append(held(ROOT / "eval/runs/v2/test_v2_guard.json", ROOT / "out/v2/test_v2_guard.log", "v2 + guard", "guard yes, rules no"))
hist.append(llmail(LM1, V2R / "sample1_v2.json", V2R / "sample1_v2.log", "v2", "LLMail sample 1", "no (after fix)"))
s2 = llmail(LM2, V2R / "sample2_v2.json", V2R / "sample2_v2.log", "v2", "LLMail sample 2", "**yes**")
s2g = llmail(LM2, V2R / "sample2_v2_guard.json", V2R / "sample2_v2_guard.log", "v2 + guard", "LLMail sample 2", "**yes**")
hist += [s2, s2g]

for h in (s2, s2g):
    name = "guard_off" if h is s2 else "guard_on"
    (CP / f"validation_llmail_v2/results_{name}.json").write_text(
        json.dumps({"summary": h["summary"], "rows": h["rows"]}, indent=2, ensure_ascii=False) + "\n")

lines = ["# v2 evaluation results", "",
         "v2 = v1 plus L5 taint rules R1/R2 (and the bare-link pattern R3); guard = ProtectAI model on (`--classifier protectai`, "
         "threshold frozen on dev). \"Blind\" only for data never used to design or tune anything.", "",
         "## Evaluation history", "",
         "| version | eval set | blind? | attacks succeed | legit lost / suppressed | clean / benign false drops | tokens sent / recv |",
         "|---|---|---|---|---|---|---|"]
lines += [f"| {h['version']} | {h['set']} | {h['blind']} | {h['succeed']} | {h['lost']} | {h['benign']} | {h['tokens']} |" for h in hist]

lines += ["", "## Which rule blocked what (dropped_task log lines)", "",
          "| run | R1 unowned in tainted message | R2 tainted forwarded block |", "|---|---|---|"]
for name, log in [("dev, R1+R2 (design set)", ROOT / "out/v2/dev_v2.log"), ("dev, R1 only", ROOT / "out/v2/dev_v2_r1only.log"),
                  ("dev, R2 only", ROOT / "out/v2/dev_v2_r2only.log"), ("held-out test", ROOT / "out/v2/test_v2.log"),
                  ("held-out test + guard", ROOT / "out/v2/test_v2_guard.log"), ("LLMail sample 1", V2R / "sample1_v2.log"),
                  ("LLMail sample 2", V2R / "sample2_v2.log"), ("LLMail sample 2 + guard", V2R / "sample2_v2_guard.log")]:
    c = rule_counts(log)
    lines.append(f"| {name} | {c['r1_unowned_in_tainted_message']} | {c['r2_tainted_forwarded_block']} |")
dev = SI.score(ROOT / "eval/injection/dev", json.loads((ROOT / "eval/runs/v2/dev_v2.json").read_text()),
               (ROOT / "out/v2/dev_v2.log").read_text())
lines.append(f"\nDev with R1+R2: {len(dev['asr'])}/{dev['n_attack']} attacks, {len(dev['suppressed'])}/{dev['n_attack']} suppressed, "
             f"{len(dev['false_drop'])}/{dev['n_clean_task']} clean tasks lost, {len(dev['clean_ok'])}/{dev['n_clean']} clean correct "
             "(same with R1 only or R2 only).")

lines += ["", "## Guard model, sentence level (from eval/guard/results.md)", ""]
g = (ROOT / "eval/guard/results.md").read_text().split("## Held-out test (scored once)")[1].strip()
lines += [g]

lines += ["", "## Sample 2 score blocks (blind)", "", "Guard off (submitted default):", "```", s2["summary"]["score_block"], "```",
          "", "Guard on:", "```", s2g["summary"]["score_block"], "```"]
for h, label in ((s2, "guard off"), (s2g, "guard on")):
    succ = [r for r in h["rows"] if r.get("outcome") == "success"]
    lines.append(f"\nSuccessful sample-2 attacks, {label}: " + (", ".join(f"{r['message_id']} ({r['stratum']}')" for r in succ) or "none"))
def guard_drops(log):
    return len(re.findall(r"^dropped_injection: \S+: [^:]+(?::\w+)?: classifier\[", Path(log).read_text(encoding="utf-8", errors="replace"), re.M))


_gt = (ROOT / "eval/guard/results.md").read_text()
rx = re.search(r"\| regex \(L1\) \| (\d+)/\d+ \| [^|]+\| (\d+)/\d+ \|", _gt).groups()
gu = re.search(r"\| regex OR guard \| (\d+)/\d+ \| [^|]+\| (\d+)/\d+ \|", _gt).groups()
gd_test, gd_s2 = guard_drops(ROOT / "out/v2/test_v2_guard.log"), guard_drops(V2R / "sample2_v2_guard.log")
r2_s1 = rule_counts(V2R / "sample1_v2.log")["r2_tainted_forwarded_block"]
a2 = s2["summary"]
lines += ["", "## Findings", "",
          f"- Blind headline (v2, guard off, LLMail sample 2): {a2['success']}/{a2['N']} attacks succeed "
          f"(95% CI {100 * a2['success_ci'][0]:.1f}% to {100 * a2['success_ci'][1]:.1f}%), {a2['kept']}/{a2['N']} legitimate tasks kept, "
          f"{a2['false_drops']}/{a2['benign_n']} benign false drops. v1 blind numbers: {hist[0]['succeed']} held-out, {hist[1]['succeed']} sample 1.",
          f"- R1/R2 fired only on the known miss (R2 dropped {r2_s1} task in sample 1, the lm024 attack task); they fired 0 times on dev, "
          "held-out and sample 2, so the sample-2 result comes from the v1 layers and the rules cost no legitimate task there.",
          f"- Sample-2 stratum A' (tool call generated) is weaker than sample 1's all-five-true stratum, which was used up; "
          f"{a2['attribution']['filter_email_url_only']}/{a2['N']} blocks rested on the email/URL pattern alone.",
          f"- The guard model removed {gd_test} sentence(s) on the held-out run and {gd_s2} on sample 2 but changed no outcome there. "
          f"It lowers reliance on the address pattern: blocks resting on it alone fall from {a2['attribution']['filter_email_url_only']} to "
          f"{s2g['summary']['attribution']['filter_email_url_only']} on sample 2. At sentence level, regex OR guard catches {gu[0]} attack "
          f"sentences against {rx[0]} for the regex, with {gu[1]} vs {rx[1]} legitimate-task flags. It stays optional, off by default.",
          "- Held-out and sample-1 reruns are after-fix and not blind; they show the fixes work on the known misses, not a fair estimate."]
(ROOT / "eval/v2_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(ROOT / "eval/runs/v2/history.json").write_text(json.dumps([{k: v for k, v in h.items() if k not in ("rows", "summary", "detail")}
                                                           for h in hist], indent=2) + "\n")
print("\n".join(lines))
