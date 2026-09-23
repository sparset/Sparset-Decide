"""Read-only model profiling; production inference code is not modified.
Run from the project root using the dedicated CUDA environment.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import types
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/decision-engine/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import transformers
import sparset_decide.engine as engine_module
from sparset_decide.engine import DecisionEngine
from sparset_decide.__main__ import load_request
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.attention import sdpa_kernel, SDPBackend

REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def gpu_status():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu",
            "--format=csv,noheader"], text=True).strip()
    except Exception as exc:
        return str(exc)

def compare(a, b):
    matching = 0
    differences = []
    for key, answer in a["answers"].items():
        pa, pb = answer["probabilities"], b["answers"][key]["probabilities"]
        matching += max(pa, key=pa.get) == max(pb, key=pb.get)
        differences.extend(abs(pa[k] - pb[k]) for k in pa)
    return {"matching_labels": matching, "fields": len(a["answers"]),
            "max_probability_difference": max(differences),
            "passes_existing_0_005_gate": matching == len(a["answers"]) and max(differences) <= .005}

def timing(rows):
    samples = [r["metadata"]["total_ms"] for r in rows]
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
            "samples_ms": samples,
            "median_preparation_ms": statistics.median(r["metadata"]["preparation_ms"] for r in rows),
            "median_prefill_ms": statistics.median(r["metadata"]["prefill_ms"] for r in rows),
            "median_branch_and_scoring_ms": statistics.median(r["metadata"]["branch_and_scoring_ms"] for r in rows)}

class Instrument:
    """Temporary stage annotations and optional device timestamps. All patches restored."""
    def __init__(self, engine, events=False):
        self.engine, self.events = engine, events
        self.records, self.originals, self.handles = [], [], []
        self.stack = defaultdict(list)

    def begin(self, name):
        marker = record_function("engine::" + name)
        marker.__enter__()
        start = torch.cuda.Event(enable_timing=True) if self.events else None
        stop = torch.cuda.Event(enable_timing=True) if self.events else None
        if start is not None:
            start.record()
        return (name, marker, start, stop, time.perf_counter())

    def end(self, state):
        name, marker, start, stop, wall = state
        if stop is not None:
            stop.record()
        elapsed = (time.perf_counter() - wall) * 1000
        marker.__exit__(None, None, None)
        self.records.append((name, start, stop, elapsed))

    def wrap(self, fn, name):
        def wrapped(*args, **kwargs):
            label = name(*args, **kwargs) if callable(name) else name
            state = self.begin(label)
            try:
                return fn(*args, **kwargs)
            finally:
                self.end(state)
        return wrapped

    def patch(self, obj, key, value):
        # Preserve inherited methods rather than leaving instance overrides behind.
        own = key in vars(obj)
        old = getattr(obj, key)
        self.originals.append((obj, key, old, own))
        setattr(obj, key, value)

    def __enter__(self):
        e = self.engine
        self.patch(e, "prepare", self.wrap(e.prepare, "prepare"))
        self.patch(e, "_hidden", self.wrap(e._hidden, lambda rows, **kw:
            "shared_prefill" if kw.get("use_cache") and kw.get("cache") is None else "branch_forward"))
        self.patch(e, "_project", self.wrap(e._project, "answer_projection"))
        self.patch(e, "_sync", self.wrap(e._sync, "synchronize"))
        original_copy = engine_module.copy
        self.patch(engine_module, "copy", types.SimpleNamespace(
            deepcopy=self.wrap(original_copy.deepcopy, "cache_deepcopy")))
        from transformers.cache_utils import DynamicCache
        self.patch(DynamicCache, "batch_repeat_interleave",
                   self.wrap(DynamicCache.batch_repeat_interleave, "cache_repeat"))
        for module in e.backbone.modules():
            name = module.__class__.__name__
            if name.endswith("Attention") or name.endswith("MLP"):
                tag = "attention" if name.endswith("Attention") else "mlp"
                def before(mod, args, tag=tag):
                    self.stack[id(mod)].append(self.begin(tag))
                def after(mod, args, output):
                    self.end(self.stack[id(mod)].pop())
                self.handles.append(module.register_forward_pre_hook(before))
                self.handles.append(module.register_forward_hook(after, always_call=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for obj, key, old, own in reversed(self.originals):
            if own:
                setattr(obj, key, old)
            else:
                delattr(obj, key)

    def summary(self):
        torch.cuda.synchronize()
        totals = defaultdict(lambda: {"calls": 0, "host_elapsed_ms": 0., "cuda_event_elapsed_ms": 0.})
        for name, start, stop, wall in self.records:
            row = totals[name]
            row["calls"] += 1
            row["host_elapsed_ms"] += wall
            if start is not None:
                row["cuda_event_elapsed_ms"] += start.elapsed_time(stop)
        return dict(totals)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("Loading cached Qwen2.5-1.5B-Instruct on CUDA", flush=True)
    engine = DecisionEngine.from_pretrained(device="cuda", revision=REVISION, local_files_only=True)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "gpu_status": gpu_status(),
        "supported_profiler_activities": [str(a) for a in torch.profiler.supported_activities()],
        "model_revision": REVISION, "dtype": str(engine.head.weight.dtype),
        "engine_sha256": hashlib.sha256(Path(engine_module.__file__).read_bytes()).hexdigest(),
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes": [
            "CUDA-event fallback provides operator device timing when CUPTI tracing is unavailable; exported trace then contains no real GPU kernel timeline.",
            "Stage/module event spans can include GPU idle time while Python enqueues work; they are not kernel-only execution time.",
            "Stage spans and module spans overlap hierarchically and must not be added together.",
            "Latency results are collected with all profiling and hooks disabled.",
            "Long-context case is synthetic performance input, not an accuracy test.",
            "Seven repeats are diagnostic samples, not a robust tail-latency study."]}
    save(out / "environment.json", environment)
    refund, refund_q = load_request(ROOT / "examples/decision_engine/refund.json")
    support, support_q = load_request(ROOT / "outputs/decision-engine/replica-presets/support_triage.request.json")
    first = next(iter(refund_q))
    long_context = ("Historical background: an earlier order was delivered successfully and its invoice was archived.\n" * 55
                    + "\nCURRENT CUSTOMER MESSAGE:\n" + refund["context"])
    cases = [
        ("refund_1", refund["context"], {first: refund_q[first]}, {first: refund["expected_labels"][first]}),
        ("refund_4", refund["context"], refund_q, refund["expected_labels"]),
        ("support_28", support["context"], support_q, {}),
        ("long_context_4", long_context, refund_q, {}),
    ]
    report = {"environment": environment, "cases": {}, "backend_probes": {}}
    for name, context, questions, expected in cases:
        print("Benchmarking " + name, flush=True)
        case_out = out / name
        case_out.mkdir()
        sequences = engine.prepare(context, questions)
        save(case_out / "request.json", {"context": context, "questions": {
            k: {"kind": q.kind, "instructions": q.instructions, "options": q.options} for k, q in questions.items()}})
        for _ in range(2):
            for mode in ("parallel", "independent"):
                engine.decide(context, questions, mode=mode)
        torch.cuda.reset_peak_memory_stats()
        rows = {"parallel": [], "independent": []}
        for i in range(args.repeats):
            for mode in (("parallel", "independent") if i % 2 == 0 else ("independent", "parallel")):
                rows[mode].append(engine.decide(context, questions, mode=mode))
            print(f"  {name}: paired run {i+1}/{args.repeats}", flush=True)
        baseline_peak = torch.cuda.max_memory_allocated()
        baseline_reserved = torch.cuda.max_memory_reserved()
        case = {"question_count": len(questions), "input_tokens": list(map(len, sequences)),
                "summary": {m: timing(r) for m, r in rows.items()},
                "paired_parity": [compare(a,b) for a,b in zip(rows["parallel"], rows["independent"])],
                "peak_allocated_bytes_unprofiled": baseline_peak,
                "peak_reserved_bytes_unprofiled": baseline_reserved, "gpu_status": gpu_status()}
        if expected:
            case["accuracy"] = {m: {"correct": sum(max(r[-1]["answers"][k]["probabilities"],
                key=r[-1]["answers"][k]["probabilities"].get) == label for k,label in expected.items()),
                "labeled_fields": len(expected)} for m,r in rows.items()}
        save(case_out / "baseline.json", {"summary": case, "results": rows})
        print("  Recording CUDA stage events: " + name, flush=True)
        with Instrument(engine, events=True) as instrument:
            event_result = engine.decide(context, questions)
        case["stage_events"] = instrument.summary()
        case["event_result_parity"] = compare(rows["parallel"][-1], event_result)
        save(case_out / "stage_events.json", case["stage_events"])
        print("  Recording operator trace: " + name, flush=True)
        with Instrument(engine):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, profile_memory=True, with_stack=False) as prof:
                traced_result = engine.decide(context, questions)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(case_out / "trace.json"))
        averages = prof.key_averages(group_by_input_shape=True)
        operators = [{"name": e.key, "count": e.count, "input_shapes": e.input_shapes,
                      "self_cpu_us": e.self_cpu_time_total, "cpu_total_us": e.cpu_time_total,
                      "self_device_us": e.self_device_time_total, "device_total_us": e.device_time_total,
                      "self_cpu_memory_bytes": e.self_cpu_memory_usage,
                      "self_device_memory_bytes": e.self_device_memory_usage} for e in averages]
        save(case_out / "operators.json", sorted(operators, key=lambda r: r["self_device_us"], reverse=True))
        (case_out / "operators.txt").write_text(averages.table(sort_by="self_device_time_total", row_limit=40), encoding="utf-8")
        case["attention_dispatch"] = sorted({e.key for e in averages if "attention" in e.key.lower() and not e.key.startswith("engine::")})
        case["trace_result_parity"] = compare(rows["parallel"][-1], traced_result)
        case["trace_has_cuda_kernel_events"] = any(str(e.device_type) == "DeviceType.CUDA" for e in prof.events())
        report["cases"][name] = case
        save(out / "summary.json", report)
        print(json.dumps({"case": name, "latency": case["summary"], "attention": case["attention_dispatch"]}), flush=True)
        del prof, averages, operators
    # Test dispatch feasibility without changing production configuration.
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        key = str(backend)
        print("Testing forced SDPA backend: " + key, flush=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with sdpa_kernel(backend):
                    engine.decide(refund["context"], refund_q)
                    results = [engine.decide(refund["context"], refund_q) for _ in range(5)]
            report["backend_probes"][key] = {"status": "ok", "timing": timing(results),
                "comparison_to_auto": compare(json.loads((out / "refund_4/baseline.json").read_text())["results"]["parallel"][-1], results[-1]),
                "result": results[-1], "warnings": [str(w.message) for w in caught]}
        except Exception as exc:
            report["backend_probes"][key] = {"status": "failed", "error": repr(exc),
                "warnings": [str(w.message) for w in caught]}
        save(out / "summary.json", report)
    print("Complete: " + str(out), flush=True)

if __name__ == "__main__":
    main()
