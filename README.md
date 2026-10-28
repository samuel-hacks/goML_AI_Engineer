# goML AI Engineer Take-Home

Vijay Samuel | 28 October 2026

Both parts of the assignment, in the folder layout the brief asks for:

```
placement/        Part 1: will a student get placed? (analysis.ipynb, predict.py, requirements.txt)
class_messages/   Part 2: class messages to a JSON task list with a local model (extract.py, NOTES.md, requirements.txt)
```

Each part also has a short written report: `placement/report.pdf` and `class_messages/report.pdf`.

## Part 1: Placement prediction

| | Result |
|---|---|
| Final model | Logistic regression (elastic-net), chosen over XGBoost, LightGBM, blends and stacking by a fixed selection rule |
| Nested CV F1 (without `status_code`) | 0.563 (precision 0.453, recall 0.754, threshold 0.25) |
| Nested CV F1 (with `status_code`) | 0.924 |
| Key decision | `status_code` behaves like a post-outcome field (92% placed when 1, 2% when 0). `predict.py` uses it only when a validity gate shows it carries the same signal as in training; otherwise it uses the model without it. |

Run:
```bash
cd placement
pip install -r requirements.txt
python predict.py --input some_students.csv --output predictions.csv
```
`analysis.ipynb` is executed and explains every check and decision (data audit, leakage analysis, model comparison, threshold choice, explainability, the gate, limitations).

## Part 2: Class messages to a task list (local model)

| | Result |
|---|---|
| Model | `qwen3:14b` via Ollama on localhost; messages never leave the machine |
| Practice folder | all six messages give the expected output, including the planted injection in m04 |
| Prompt-injection defense | layered: input filter, spotlighting, instruction rules, JSON schema, post-model checks, field validator |
| Held-out attacks (own set, blind) | v1: 1/28 succeeded |
| Microsoft LLMail-Inject (external, blind) | v1: 1/100; v2 on a fresh sample: 0/100 (95% CI 0.0% to 3.7%), 100/100 legitimate tasks kept |
| Hallucination control | owner and due must appear in the message; 0 invented values in evaluation |
| Repeatability | two runs give byte-identical output |

Run:
```bash
# install Ollama (https://ollama.com), then:
ollama pull qwen3:14b
cd class_messages
pip install -r requirements.txt
python extract.py --inbox messages --output tasks.json
```
`NOTES.md` covers model, tokens, prompt, determinism, hallucination, format and validation, errors and guardrails. An optional local guard model (ProtectAI DeBERTa, Apache-2.0) is evaluated in `eval/guard/` and is off by default.

## Credits
Datasets and models used for evaluation: Microsoft LLMail-Inject (MIT), ProtectAI `deberta-v3-base-prompt-injection-v2` (Apache-2.0), Qwen3 (Alibaba), Llama 3.1 (Meta) as a comparison model. Full references are in the two reports.
