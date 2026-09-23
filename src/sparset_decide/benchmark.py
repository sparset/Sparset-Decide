"""Paired, same-model benchmark. Published M4 Max results are not this baseline."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers

from .__main__ import add_model_arguments, build_engine, load_request


def _label(answer):
    return max(answer["probabilities"], key=answer["probabilities"].get)


@torch.inference_mode()
def json_baseline(engine, data, questions, max_new_tokens):
    """Less demanding baseline: emit labels, not distributions. Report that fact."""
    engine._sync()
    start = time.perf_counter()
    specifications = {key: {"question": q.instructions, "options": dict(q.options)}
                      for key, q in questions.items()}
    prompt = engine.tokenizer.apply_chat_template([
        {"role": "system", "content": "Classify the supplied context. Return only one JSON object mapping each field ID to one allowed option label string. Include every field. Do not explain."},
        {"role": "user", "content": json.dumps({"context": data["context"], "fields": specifications}, ensure_ascii=False)},
    ], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if len(prompt) > engine.max_input_tokens:
        return {"status": "skipped_input_too_long", "input_tokens": len(prompt), "total_ms": None}
    ids = torch.tensor([prompt], device=engine.device)
    output = engine.model.generate(
        input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
        max_new_tokens=max_new_tokens, pad_token_id=engine.tokenizer.eos_token_id,
        use_cache=True,
    )
    engine._sync()
    total_ms = (time.perf_counter() - start) * 1000
    tokens = output[0, len(prompt):].cpu().tolist()
    text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
    allowed_eos = engine.model.generation_config.eos_token_id
    allowed_eos = allowed_eos if isinstance(allowed_eos, list) else [allowed_eos]
    truncated = len(tokens) >= max_new_tokens and (not tokens or tokens[-1] not in allowed_eos)
    labels = None
    error = None
    try:
        labels = json.loads(text)
        if not isinstance(labels, dict) or set(labels) != set(questions):
            raise ValueError("Output fields do not exactly match requested fields")
        for key, q in questions.items():
            if not isinstance(labels[key], str) or labels[key] not in dict(q.options):
                raise ValueError(f"Invalid option label for {key}")
    except (ValueError, TypeError) as exc:
        error = str(exc)
    return {"status": "truncated" if truncated else "invalid" if error else "valid",
            "input_tokens": len(prompt), "generated_tokens": len(tokens),
            "total_ms": total_ms, "labels": labels, "error": error, "raw_text": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--include-json", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--probability-atol", type=float, default=0.005)
    add_model_arguments(parser)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.max_new_tokens < 1:
        parser.error("repeats/max-new-tokens must be positive and warmup nonnegative")
    if not 0 <= args.probability_atol < 1:
        parser.error("probability-atol must be in [0, 1)")
    data, questions = load_request(args.input)
    expected = data.get("expected_labels", {})
    if expected and (set(expected) - set(questions) or any(
        label not in dict(questions[key].options) for key, label in expected.items()
    )):
        raise ValueError("expected_labels must contain valid option labels")
    engine = build_engine(args)
    for _ in range(args.warmup):
        for mode in ("parallel", "independent"):
            engine.decide(data["context"], questions, mode=mode)
        if args.include_json:
            json_baseline(engine, data, questions, args.max_new_tokens)
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    records = {"parallel": [], "independent": []}
    json_records = []
    for repeat in range(args.repeats):
        # Alternate order to reduce systematic warmup/thermal bias.
        modes = ("parallel", "independent") if repeat % 2 == 0 else ("independent", "parallel")
        for mode in modes:
            records[mode].append(engine.decide(data["context"], questions, mode=mode))
        if args.include_json:
            json_records.append(json_baseline(engine, data, questions, args.max_new_tokens))
        print(f"Completed paired repetition {repeat + 1}/{args.repeats}", flush=True)
    deltas = []
    matching = 0
    for parallel, independent in zip(records["parallel"], records["independent"]):
        for key in questions:
            a, b = parallel["answers"][key], independent["answers"][key]
            deltas.extend(abs(a["probabilities"][label] - b["probabilities"][label]) for label in a["probabilities"])
            matching += _label(a) == _label(b)
    summary = {}
    for mode, rows in records.items():
        times = [row["metadata"]["total_ms"] for row in rows]
        summary[mode] = {"median_ms": statistics.median(times), "min_ms": min(times),
                         "max_ms": max(times), "samples_ms": times,
                         "questions_per_second_at_median": len(questions) * 1000 / statistics.median(times)}
    quality = None
    if expected:
        quality = {mode: {"correct": sum(_label(rows[-1]["answers"][key]) == label for key, label in expected.items()),
                          "labeled_fields": len(expected)} for mode, rows in records.items()}
        for row in json_records:
            row["quality"] = {
                "correct": sum(row["labels"].get(key) == label for key, label in expected.items())
                if row["status"] == "valid" else 0,
                "labeled_fields": len(expected),
            }
    valid_json_times = [row["total_ms"] for row in json_records if row["status"] == "valid"]
    json_median = statistics.median(valid_json_times) if valid_json_times else None
    report = {
        "description": "Same-model, same-device direct-scoring comparison; not a reproduction of the HF replica's M4 Max timings.",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source": data.get("source"), "model": args.model,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
                        "device": str(engine.device),
                        "gpu": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None},
        "warmup_runs": args.warmup, "repeats": args.repeats, "summary": summary,
        "speedup_vs_independent": summary["independent"]["median_ms"] / summary["parallel"]["median_ms"],
        "parity": {"max_probability_difference": max(deltas), "tolerance": args.probability_atol,
                   "matching_decisions": matching, "total_decisions": len(questions) * args.repeats,
                   "passed": max(deltas) <= args.probability_atol and matching == len(questions) * args.repeats},
        "quality": quality,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(engine.device) if engine.device.type == "cuda" else None,
        "results": records,
        "json_baseline": {"output_contract": "Labels only; no probability distributions; different prompt from direct scoring.",
                          "valid_median_ms": json_median,
                          "speedup_vs_valid_json": json_median / summary["parallel"]["median_ms"] if json_median else None,
                          "valid_runs": sum(row["status"] == "valid" for row in json_records), "runs": json_records},
        "limitations": ["Probabilities have not been calibrated.",
                        "Tiny smoke examples do not establish general accuracy.",
                        "No direct benchmark of the original HF engine has been run by this command.",
                        "Published speedup factors cannot be compared across different devices or precision."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "parity": report["parity"], "quality": quality}, indent=2))
    raise SystemExit(0 if report["parity"]["passed"] else 2)


if __name__ == "__main__":
    main()
