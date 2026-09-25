"""Summarize measured implementations, retaining negative results and cold costs."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/optimization-20260918'
names=['cache','graphs','rmsnorm','swiglu','rope','fused-initial256','fused','combined']
data={n:json.loads((OUT/(n+'.json')).read_text())for n in names if (OUT/(n+'.json')).exists()}
lines=['# Implemented inference optimizations — 18 September 2026','',
'CUDA graph replay, cache optimizations and three custom Triton kernels are implemented in the dedicated Inference Engine repository. Changes are uncommitted. No files were intentionally changed in the Distillation project.','',
'## Paired end-to-end results','',
'Each cell is original-engine median → variant median in milliseconds, followed by speed ratio. Greater than 1x is faster. Seven measured pairs per workload after a cold call and two warmups; baseline/variant order alternated. These timings include tokenization, inference, scoring, CPU transfer and result assembly, but exclude model loading and warmup/capture/JIT. Raw JSON includes every timing and cold-request cost. Ratios from different rows must not be multiplied.','',
'| Change | One question | Four questions | 28 questions | Long context, four questions |','|---|---:|---:|---:|---:|']
for name,d in data.items():
 cells=[]
 for case in ('refund_1','refund_4','support_28','long_context_4'):
  c=d['cases'].get(case)
  cells.append('pending'if c is None else f"{c['median_ms']['baseline']:.0f} → {c['median_ms']['variant']:.0f} ({c['speedup']:.2f}x)")
 lines.append('| '+name+' | '+' | '.join(cells)+' |')
lines+=['','cache = metadata-only prefix forks plus cached selected FP32 answer-head rows. graphs = cache changes plus bounded CUDA graphs. rmsnorm/swiglu/rope each enable that one custom kernel plus cache changes. fused enables all three kernels plus cache changes, without graphs. combined enables all three kernels, cache changes and bounded graphs. All are compared with the original engine, rather than with the immediately preceding table row. The individual swiglu/rope rows and fused-initial256 use the initial 256-element launch blocks. The final fused and combined rows use the tuned 512-element blocks.','',
'## Numerical effects and labeled smoke case','',
'Probability drift below is the maximum absolute difference across the original and changed-input requests; percentage points are absolute probability differences times 100. Label agreement does not prove task accuracy. Values above the old 0.005 probability gate are reported rather than automatically rejected, following the requested emphasis on useful speed/quality tradeoffs.','',
'| Change | Largest probability drift (percentage points) | Original + changed-input label agreement | Refund labeled smoke: original / variant |','|---|---:|---:|---:|']
expected=json.loads((ROOT/'examples/decision_engine/refund.json').read_text())['expected_labels']
for name,d in data.items():
 comparisons=[c[k]for c in d['cases'].values()for k in ('parity','changed_input_parity')]
 diff=max(x['max_probability_difference']for x in comparisons);matched=sum(x['matching_labels']for x in comparisons);total=sum(x['fields']for x in comparisons)
 score='pending'
 if 'refund_4'in d['cases']:
  c=d['cases']['refund_4'];counts=[]
  for v in ('baseline','variant'):
   answers=c['rows'][v][0]['answers'];counts.append(sum(max(answers[k]['probabilities'],key=answers[k]['probabilities'].get)==label for k,label in expected.items()))
  score=f'{counts[0]}/4 / {counts[1]}/4'
 lines.append(f'| {name} | {diff*100:.4f} | {matched}/{total} | {score} |')
lines+=['',
'Four labeled fields cannot establish a 0.5% accuracy change. The support workload has no verified ground-truth labels. We did not train, calibrate or quantize the model. A representative labeled evaluation set remains necessary before claiming a sub-percent task-accuracy tradeoff.','',
'## Practical choice and the changed decision','', 'For repeated short inputs with matching shapes, the final combined path measured 1.78x (one question) and 1.69x (four questions). No tested end-to-end case reached 2x. For the larger workloads, tuned fusion without graphs gave roughly 11–17% gains across the suite and preserved all selected labels; adding graphs yielded little additional benefit in the 28-question case. These are local measurements, not universal speed guarantees.','', 'The combined support run changed contract_penalty_applicable from yes=0.5112286210 (original) to yes=0.4974556267 (combined), crossing 50%. This accounts for the one label disagreement. It has no verified ground-truth label, so it is neither a demonstrated accuracy loss nor a demonstrated improvement. The four labeled refund fields remained 3/4 correct.','', 'Graphs are exact-shape, not automatic length buckets: a stream of new lengths can incur repeated capture costs and may be slower. Leave graphs off for one-shot or highly variable-shape workloads unless measured otherwise. Cold combined request costs in this run were approximately 1.6–5.2 seconds, excluding model loading.','', '## Cold cost, memory and fallback','',
'Graphs and Triton kernels need capture/compilation before warm reuse. Reusing one engine instance matters: a one-shot CLI invocation does not realize steady-state graph speedups. Graph input IDs, positions, masks, last-token indices and cached prefix tensors are refreshed on every request.','',
'| Variant / workload | First variant request ms | Warm peak PyTorch allocation MiB | Final graph resident entries / cumulative fallbacks |','|---|---:|---:|---:|']
for name in ('graphs','fused','combined'):
 if name not in data:continue
 for case,c in data[name]['cases'].items():
  meta=c['rows']['variant'][-1]['metadata'].get('optimizations',{});stats=meta.get('graph_stats',{})
  lines.append(f"| {name} / {case} | {c['cold']['variant']['external_ms']:.0f} | {c['peak_allocated_bytes']/2**20:.1f} | {meta.get('graph_entries',0)} / {stats.get('fallbacks',0)} |")
lines+=['',
'Peak allocation is measured over warm paired calls in a process sharing one set of model weights. It excludes some CUDA/runtime memory and does not report peak compilation/capture allocation. Baseline calls in a graph comparison coexist with the retained graph buffers. The original engine class shares the same weights; fusion adapters are inactive during reference calls (the forwarding wrappers remain installed).','',
'The first graph implementation exposed an indexing-buffer lifetime bug during replay. Keeping that buffer alive fixed it, and changed-input/eviction tests now cover the issue. An initial 384 MiB graph policy hit VRAM pressure while capturing multiple support-branch shapes; that run was stopped and its completed small-workload results are retained as graphs-initial-partial.json, not used as the main result.','',
'The implemented default is a 192 MiB retained-allocation budget, at most four resident graphs, and conservative pre-capture rejection based on prefix KV/scratch needs. Unsupported or oversized shapes fall back to ordinary execution and failed shapes are remembered. The total model memory and transient capture peak are not capped by this budget. Long inputs and many branches can consequently see small gains or regressions; graphs remain opt-in. Cache improvements are enabled by default and individually disableable.','',
'## Kernel launch-size selection','', 'kernel-launch-sweep.json records GPU-only operation timing for blocks of 256, 512, 1024 and 2048 elements. A 512-element block was a conservative improvement across the tested SwiGLU and RoPE shapes and was selected for the final code. Kernel-only speed ratios are not full-model speed ratios. The final fused and combined stages remeasure the complete request after this change.','', '## Implementation and validation','',
'- src/subset/engine.py: configuration, cache fork/head-row reuse, graph dispatch, model-scoped serialization and metadata.','- src/subset/cuda_graphs.py: exact-shape input buffers, capture/replay, bounded storage, memory admission and fallback.','- src/subset/kernels.py: custom Triton RMSNorm, SiLU-plus-multiplication (SwiGLU), and query/key RoPE. Intermediate low-precision rounding is retained where possible.','- src/subset/fusion.py: model-local adapters; no global Transformers patch. Fusion is validated for dense Qwen2 with Transformers 5.5.4.','- tests/test_optimizations.py: cache isolation/invalidation, changed-input graph replay, masks, fallback, eviction, CPU fallback, kernel numerical checks and all-fusion graph integration.','- docs/optimizations.md: API/CLI usage, dependency requirements, memory behavior and limitations.','',
'Full model: Qwen/Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, FP16 backbone / FP32 selected scoring head. RTX 3050 Laptop 4 GB, Ubuntu/WSL, PyTorch 2.6.0+cu124, Transformers 5.5.4, Triton 3.2.0. Custom kernels compiled using an isolated Zig 0.13.0 compiler and matching Python headers. GPU runs are sequential to avoid benchmark interference. Laptop timing noise remains; the single-question case is particularly variable.','']
if (OUT/'tests.json').exists():
 t=json.loads((OUT/'tests.json').read_text());lines.append(f"Final verification: {t['tests_run']} tests; success={t['successful']}; skipped={len(t['skipped'])}.")
else:lines.append('Final test-summary artifact pending; earlier complete suite passed 16 tests before adding two focused fallback/mask checks.')
lines+=['','## Reproduce','',
'Use scripts/run_optimization.sh --stage cache (or graphs, rmsnorm, swiglu, rope, fused, combined) in the configured local Linux environment. scripts/run_optimization_tests.py runs the verification suite without downloading models. scripts/summarize_optimizations.py regenerates this report from saved results. The saved baseline_engine.py artifact contains the pre-change engine; its SHA256 is '+hashlib.sha256((OUT/'baseline_engine.py').read_bytes()).hexdigest()+'. The benchmark uses that saved reference rather than a second physical copy of the model weights.','',
'Raw stage JSON files contain actual model probability outputs, not fabricated example responses. No comparison against the external Hugging Face replica was run in this implementation task.','',
'Technical background: [PyTorch CUDA graph memory and lifetime requirements](https://docs.pytorch.org/docs/main/notes/cuda.html), [Triton fused normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).','']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in (ROOT/'src/subset').glob('*.py')}
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT/'REPORT.md')
