"""Diagnostic backend routing experiment only; no production files/settings change."""
import argparse
import contextlib
import json
from pathlib import Path
import profile_engine as harness
import torch
import transformers.integrations.sdpa_attention as sdpa
from torch.profiler import profile, ProfilerActivity
from subset.engine import DecisionEngine
from subset.__main__ import load_request

@contextlib.contextmanager
def routing(expand_kv):
    original = sdpa.use_gqa_in_sdpa
    try:
        if expand_kv:
            sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
        yield
    finally:
        sdpa.use_gqa_in_sdpa = original

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    capabilities = {}
    for name in ("is_flash_attention_available", "flash_sdp_enabled",
                 "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        fn = getattr(torch.backends.cuda, name, None)
        capabilities[name] = fn() if fn else None
    harness.save(out / "capabilities.json", capabilities)
    print("Backend build capabilities: " + json.dumps(capabilities), flush=True)
    engine = DecisionEngine.from_pretrained(device="cuda", revision=harness.REVISION, local_files_only=True)
    refund, rq = load_request(harness.ROOT / "examples/decision_engine/refund.json")
    support, sq = load_request(harness.ROOT / "outputs/decision-engine/replica-presets/support_triage.request.json")
    first = next(iter(rq))
    long_context = ("Historical background: an earlier order was delivered successfully and its invoice was archived.\n" * 55
                    + "\nCURRENT CUSTOMER MESSAGE:\n" + refund["context"])
    cases = [("refund_1", refund["context"], {first: rq[first]}),
             ("refund_4", refund["context"], rq),
             ("support_28", support["context"], sq),
             ("long_context_4", long_context, rq)]
    report = {"capabilities": capabilities, "cases": {}, "note":
        "Temporary in-process diagnostic: force Transformers to expand grouped K/V heads before SDPA instead of passing enable_gqa=True. "
        "Default automatic backend selection remains enabled. No persistent configuration or model weights change. "
        "Timings are unprofiled; only three paired samples per workload and two warmups per variant."}
    original = sdpa.use_gqa_in_sdpa
    for name, context, questions in cases:
        print("Diagnostic paired runs: " + name, flush=True)
        rows = {"default": [], "expanded_kv": []}
        for _ in range(2):
            for variant in rows:
                with routing(variant == "expanded_kv"):
                    engine.decide(context, questions)
        for i in range(3):
            for variant in (("default", "expanded_kv") if i % 2 == 0 else ("expanded_kv", "default")):
                with routing(variant == "expanded_kv"):
                    torch.cuda.reset_peak_memory_stats()
                    result = engine.decide(context, questions)
                    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    rows[variant].append(result)
            print(f"  {name}: {i+1}/3", flush=True)
        case = {"summary": {k: harness.timing(v) for k,v in rows.items()},
                "parity": [harness.compare(a,b) for a,b in zip(rows["default"], rows["expanded_kv"])],
                "peak_allocated_bytes": {k:max(r["peak_allocated_bytes"] for r in v) for k,v in rows.items()},
                "results": rows}
        report["cases"][name] = case
        harness.save(out / "summary.json", report)
        print(json.dumps({"case": name, "timings": case["summary"], "parity":case["parity"]}), flush=True)
    # Collect traces only after all unprofiled timing comparisons.
    case = report["cases"]["refund_4"]
    context, questions = refund["context"], rq
    case["dispatch"] = {}
    for variant in rows:
        with routing(variant == "expanded_kv"):
            with harness.Instrument(engine):
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    result = engine.decide(context, questions)
                    torch.cuda.synchronize()
        prof.export_chrome_trace(str(out / (variant + "_dispatch_trace.json")))
        case["dispatch"][variant] = {
            e.key: e.count for e in prof.key_averages()
            if "attention" in e.key.lower() and not e.key.startswith("engine::")}
    harness.save(out / "summary.json", report)
    report["routing_restored"] = sdpa.use_gqa_in_sdpa is original
    harness.save(out / "summary.json", report)
    print("Diagnostic complete; original routing restored: " + str(report["routing_restored"]), flush=True)

if __name__ == "__main__":
    main()
