"""Per-item side-by-side: baseline (b) vs the `correct` fine-tune, on BOTH eval groups.

NB5 §5 ranks the fine-tune's own target predictions, which says nothing when every
target item scores 1.0 — and hides the place the fine-tune actually loses (regression).
This script regenerates both models' outputs with exactly the NB2/NB5 settings
(greedy; target: (b) with OPTIMIZED_PROMPT, FT with NAIVE_PROMPT; regression: no
system prompt, 96 new tokens) and writes `results/side_by_side.json`, so the report's
qualitative examples — wins AND losses — come from a file, not from memory.

    python scripts/side_by_side.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report  # noqa: E402
from labkit.config import get_tier  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def field_hits(pred: str, label: dict) -> dict:
    obj = ev._parse_json_loose(pred) or {}
    return {k: ev.normalize(str(obj.get(k, ""))) == ev.normalize(str(label[k]))
            for k in ev.TRIAGE_KEYS}


def run(model, tok, system_target, label, target, regression):
    t, _ = generate.generate_batch(model, tok, [r["input"] for r in target],
                                   system=system_target, label=f"{label}/target")
    g, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression],
                                   system=None, max_new_tokens=96, label=f"{label}/regression")
    return t, g


def main() -> None:
    from peft import PeftModel

    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")

    model, tok = generate.load_base(tier)
    b_t, b_g = run(model, tok, generate.OPTIMIZED_PROMPT, "(b)", target, regression)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
    model.eval()
    f_t, f_g = run(model, tok, generate.NAIVE_PROMPT, "ft", target, regression)

    tgt_rows = []
    for i, (r, pb, pf) in enumerate(zip(target, b_t, f_t)):
        tgt_rows.append({
            "i": i, "ticket": r["input"], "label": r["label"],
            "b_pred": pb, "b_score": round(ev.triage_field_accuracy(pb, r["label"]), 2),
            "b_fields": field_hits(pb, r["label"]),
            "ft_pred": pf, "ft_score": round(ev.triage_field_accuracy(pf, r["label"]), 2),
        })
    reg_rows = []
    for i, (r, pb, pf) in enumerate(zip(regression, b_g, f_g)):
        reg_rows.append({
            "i": i, "question": r["instruction"], "keywords": r["keywords"],
            "b_pred": pb, "b_recall": round(ev.keyword_recall(pb, r["keywords"]), 2),
            "ft_pred": pf, "ft_recall": round(ev.keyword_recall(pf, r["keywords"]), 2),
        })

    n = len(tgt_rows)
    summary = {
        "b_target": round(sum(x["b_score"] for x in tgt_rows) / n, 4),
        "ft_target": round(sum(x["ft_score"] for x in tgt_rows) / n, 4),
        "b_field_acc": {k: round(sum(x["b_fields"][k] for x in tgt_rows) / n, 4)
                        for k in ev.TRIAGE_KEYS},
        "b_regression": round(sum(x["b_recall"] for x in reg_rows) / len(reg_rows), 4),
        "ft_regression": round(sum(x["ft_recall"] for x in reg_rows) / len(reg_rows), 4),
        "target_ft_wins": sum(x["ft_score"] > x["b_score"] for x in tgt_rows),
        "target_ft_losses": sum(x["ft_score"] < x["b_score"] for x in tgt_rows),
        "regression_ft_wins": sum(x["ft_recall"] > x["b_recall"] for x in reg_rows),
        "regression_ft_losses": sum(x["ft_recall"] < x["b_recall"] for x in reg_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    report.write_json({"summary": summary, "target": tgt_rows, "regression": reg_rows},
                      "side_by_side.json", results_dir=ROOT / "results")


if __name__ == "__main__":
    main()
