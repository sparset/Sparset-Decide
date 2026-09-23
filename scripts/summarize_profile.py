"""Summarize saved profiling artifacts without loading a model."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/decision-engine/profiling-20260918"
r = json.loads((OUT / "summary.json").read_text())
d = json.loads((OUT / "backend-diagnostic/summary.json").read_text())
assert len(r["cases"]) == 4 and len(d["cases"]) == 4 and d["routing_restored"]
unchanged = hashlib.sha256((ROOT / "src/sparset_decide/engine.py").read_bytes()).hexdigest() == r["environment"]["engine_sha256"]
assert unchanged
names = {"refund_1": "1 question", "refund_4": "4 refund questions", "support_28": "28 support questions", "long_context_4": "4 questions, longer context"}
lines = [
"# Qwen decision-engine profiling — 18 September 2026", "",
"Profiling completed on the RTX 3050 4 GB Laptop GPU. Production inference code and model weights were not changed. "
"The additional routing experiment was temporary and restored the original function.", "",
"## Findings", "",
"1. **FlashAttention is absent from the installed PyTorch build.** "
"`flash_sdp_enabled()` is true, but `is_flash_attention_available()` is false. "
"Forcing FlashAttention fails with `Torch was not compiled with flash attention`.",
"2. **Attention backend selection differs by stage.** Single-question forwards and shared prefills use "
"`aten::_scaled_dot_product_attention_math`. Cached question branches use "
"`aten::_scaled_dot_product_efficient_attention` / `aten::_efficient_attention_forward`. "
"An SDPA setting alone does not guarantee FlashAttention.",
"3. **Grouped attention heads explain the observed prefill fallback.** "
"Qwen supplies 12 query heads and 2 key/value heads. The installed Transformers integration uses native "
"SDPA grouped-query attention for the unmasked prefill; the available efficient kernel rejects unequal head counts. "
"Masked branches expand K/V heads first and can use the efficient kernel.",
"4. **A compatibility experiment enabled the existing fused kernel for prefill.** "
"It explicitly expands K/V heads before SDPA, with no weight changes. It helped the long-context workload, "
"but gains varied by workload and the support workload exceeded the numerical tolerance.",
"5. **Forward execution dominates.** Attention and MLP processing both matter. Cache deep-copy/repeat is a "
"secondary target in the 28-question workload; answer projection is relatively small.",
"6. **Timings are noisy and session-dependent.** The main suite's later timings were much slower than the "
"fresh-process diagnostic. Both sets are retained. One CPU snapshot showed 90% usage; the GPU reported an active "
"35 W software power cap in another snapshot. These observations do not prove the cause of every outlier.", "",
"## Environment and method", "",
f"- PyTorch {r['environment']['torch']}; CUDA runtime {r['environment']['cuda_runtime']}; "
f"Transformers {r['environment']['transformers']}; Python {r['environment']['python']}.",
"- Qwen/Qwen2.5-1.5B-Instruct; FP16 backbone, FP32 selected answer projection; branch batch size 8.",
f"- Model revision: `{r['environment']['model_revision']}`.",
"- Main suite: two warmups per mode, seven alternating paired repetitions per workload. "
"Timings exclude model loading and are recorded with profiling/hooks disabled. Each workload's trace follows its timing runs.",
"- Diagnostic: a fresh process, two warmups per variant, three alternating paired repetitions per workload; "
"all timings finish before any trace capture.",
"- Longer context is synthetic historical text followed by the refund message. It is a performance input, not an accuracy test.",
"- Windows Kineto exposes CPU-only tracing. CUDA-event fallback supplies operator timing estimates, and separate CUDA events "
"measure stage spans. Exported JSON traces contain CPU/operator ranges, not native GPU kernel timelines.",
"- CUDA-event spans can include CPU submission gaps. Nested module/stage spans overlap. They are not pure kernel times.",
"- Profiling itself adds overhead. Use the unprofiled comparisons for latency; use traces to identify executed operations.",
"- Native GPU tracing with Nsight Systems/Compute or a CUDA-trace-capable setup is still required to inspect kernel names, "
"occupancy, hardware counters and exact kernel-versus-launch overhead.", "",
"## Main unprofiled measurements", "",
"| Workload | Tokens/question | Parallel median (range), ms | Independent median (range), ms | Peak allocated MiB |",
"|---|---:|---:|---:|---:|"]
for key,c in r["cases"].items():
    a,b=c["summary"]["parallel"],c["summary"]["independent"]
    lens=c["input_tokens"]
    lines.append(f"| {names[key]} | {min(lens)}–{max(lens)} | {a['median_ms']:.1f} ({a['min_ms']:.1f}–{a['max_ms']:.1f}) | "
                 f"{b['median_ms']:.1f} ({b['min_ms']:.1f}–{b['max_ms']:.1f}) | {c['peak_allocated_bytes_unprofiled']/2**20:.1f} |")
lines += ["", "The one-question modes execute the same algorithm; their timing difference is variability, not a branching benefit. "
          "These are local diagnostics, not a comparison with Jev or the HF replica.", "",
          "## Fresh-process backend-routing diagnostic", "",
          "Default routing and expanded-K/V routing use the same model, device, precision, and automatic SDPA backend selection. "
          "Expanding K/V heads does not change the model architecture; it changes how equivalent attention inputs reach the kernel. "
          "Different kernels still produce numerical differences.", "",
          "| Workload | Default median, ms | Expanded K/V median, ms | Ratio of medians | Largest probability difference | Labels matched |",
          "|---|---:|---:|---:|---:|---:|"]
for key,c in d["cases"].items():
    a,b=c["summary"]["default"]["median_ms"],c["summary"]["expanded_kv"]["median_ms"]
    delta=max(p["max_probability_difference"] for p in c["parity"])
    matches=sum(p["matching_labels"] for p in c["parity"])
    fields=sum(p["fields"] for p in c["parity"])
    lines.append(f"| {names[key]} | {a:.1f} | {b:.1f} | {a/b:.2f}× | {delta:.8f} | {matches}/{fields} |")
lines += ["", "**Interpretation:** the long-context case improved from 915 to 661 ms, roughly 28% less latency. "
          "The short four-question and 28-question medians changed little. Three samples do not establish a general speedup. "
          "All selected labels matched, but the support workload's 0.00767 probability change fails the existing 0.005 gate. "
          "The experiment is not ready to become a production default.", "",
          "Observed four-question dispatch counts:", "", "```json",
          json.dumps(d["cases"]["refund_4"]["dispatch"], indent=2), "```", "",
          "## Main numerical checks", ""]
for key,c in r["cases"].items():
    pairs=c["paired_parity"]
    delta=max(p["max_probability_difference"] for p in pairs)
    matches=sum(p["matching_labels"] for p in pairs)
    fields=sum(p["fields"] for p in pairs)
    lines.append(f"- {names[key]}: {matches}/{fields} labels agree between parallel and independent execution; "
                 f"max probability difference {delta:.8f}; "
                 f"existing 0.005 gate {'passes' if all(p['passes_existing_0_005_gate'] for p in pairs) else 'fails'}.")
lines += ["", "The labeled four-question refund example remains 3/4 correct. Agreement does not establish accuracy "
          "or calibration. The support scenario has no verified labels.", "",
          "## Instrumented stage spans", "",
          "These are CUDA-event elapsed spans from a separately instrumented request, including possible CPU submission gaps. "
          "Attention and MLP are contained inside prefill/branch spans. Do not sum nested columns or compare these directly "
          "with unprofiled medians.", "",
          "| Workload | Prefill ms | Branch forwards ms | Cache clone/repeat ms | Answer projection ms | Attention modules ms | MLP modules ms |",
          "|---|---:|---:|---:|---:|---:|---:|"]
for key,c in r["cases"].items():
    stages=c["stage_events"]
    def ms(k): return stages.get(k,{}).get("cuda_event_elapsed_ms",0)
    lines.append(f"| {names[key]} | {ms('shared_prefill'):.1f} | {ms('branch_forward'):.1f} | "
                 f"{ms('cache_deepcopy')+ms('cache_repeat'):.1f} | {ms('answer_projection'):.1f} | "
                 f"{ms('attention'):.1f} | {ms('mlp'):.1f} |")
lines += ["", "## Recommended order", "",
          "1. Address prefill attention dispatch: evaluate a supported expanded-K/V compatibility path or a runtime "
          "that actually includes FlashAttention. Recheck numerical differences against a higher-precision reference "
          "before changing defaults.",
          "2. Stabilize measurements with host-load/clock tracking and more unprofiled repetitions. "
          "Keep cold-start and warmed latency separate and retain outliers.",
          "3. Reduce cache deep copies and repeated allocations across branch chunks. Then evaluate branch-length grouping "
          "and batch size. Preserve branch isolation and compare both latency and peak memory.",
          "4. Obtain native GPU traces before selecting custom kernels. Investigate fusion/compilation for small operations "
          "only when launch or memory overhead is demonstrated. Existing MLP matrix multiplications already use libraries.",
          "5. Evaluate task accuracy separately. Faster execution does not repair the incorrect refund answer or calibrate probabilities.", "",
          "## Artifacts", "",
          "- [Main summary](summary.json), [environment](environment.json), "
          "[diagnostic summary](backend-diagnostic/summary.json), [backend capabilities](backend-diagnostic/capabilities.json).",
          "- Each workload folder contains the exact request, baseline results, stage events, operator tables, and a Chrome-format trace.",
          "- Traces contain CPU/operator activity; native CUDA kernel timelines are absent on this build.",
          "- Production engine SHA-256 matches the pre-run value; temporary backend routing was restored.", "",
          "## Reproduction", "",
          "Run from the Inference Engine directory, choosing new output folders. The scripts use cached model weights and make no network requests.", "",
          "```powershell",
          ".cache/decision-engine/venv/Scripts/python.exe -u scripts/profile_engine.py --output outputs/decision-engine/profiling-new --repeats 7",
          ".cache/decision-engine/venv/Scripts/python.exe -u scripts/profile_backend_diagnostic.py --output outputs/decision-engine/profiling-new/backend-diagnostic",
          "```", ""]
(OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"report": str(OUT / "REPORT.md"), "production_engine_unchanged": unchanged,
                  "diagnostic_routing_restored": d["routing_restored"]}, indent=2))
