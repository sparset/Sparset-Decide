"""Aggregate the completed profiling artifacts without rerunning the model."""
import json, collections, hashlib, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/decision-engine/profiling-20260918'
def read(p): return json.loads((OUT/p).read_text(encoding='utf-8-sig'))
def save(p,d): (OUT/p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
d=read('cupti-model/summary.json'); groups=collections.defaultdict(lambda:{'count':0,'duration_ms':0.0}); names=collections.defaultdict(lambda:{'count':0,'duration_ms':0.0})
for k in d['kernel_records']:
 n=k['name'].lower()
 g='Attention' if any(s in n for s in ('flash','fmha','attention')) else 'Matrix multiplication' if any(s in n for s in ('gemm','gemv')) else 'Copy / indexing / layout' if any(s in n for s in ('copy','indexselect','catarray','fill','transpose')) else 'Reductions' if any(s in n for s in ('reduce','softmax')) else 'Elementwise' if 'elementwise' in n else 'Other'
 for entry in (groups[g],names[k['name']]): entry['count']+=1;entry['duration_ms']+=k['duration_us']/1000
intervals=sorted((k['start_ns'],k['end_ns']) for k in d['kernel_records']); union=0;s,e=intervals[0]
for a,b in intervals[1:]:
 if a>e:union+=e-s;s,e=a,b
 else:e=max(e,b)
union+=e-s
summary={'scope':'One warmed four-question refund request; direct CUPTI concurrent-kernel activity. Grouping is heuristic by kernel name, not module attribution.', 'kernel_count':len(intervals),'dropped_records':d['dropped'],'errors':d['errors'],'profiled_wall_ms':d['profiled_wall_ms'],'profiled_cuda_event_ms':d['profiled_cuda_event_ms'],'kernel_duration_sum_ms':sum(k['duration_ms'] for k in groups.values()),'kernel_interval_union_ms':union/1e6,'kernel_span_ms':(max(b for a,b in intervals)-intervals[0][0])/1e6,'groups':dict(groups),'kernels_by_duration':[dict(name=n,**v)for n,v in sorted(names.items(),key=lambda x:-x[1]['duration_ms'])]}
save('cupti-model/kernel-analysis.json',summary)
origin=intervals[0][0]
trace={'displayTimeUnit':'ms','otherData':{'scope':'GPU-only relative CUPTI clock; not aligned to separate CPU/Kineto trace'},'traceEvents':[{'name':k['name'],'cat':'kernel','ph':'X','pid':0,'tid':k['stream'],'ts':(k['start_ns']-origin)/1000,'dur':k['duration_us'],'args':{a:k[a]for a in ('correlation','registers_per_thread','grid','block')}}for k in d['kernel_records']]}
save('cupti-model/gpu-trace.json',trace)
p=read('deep-linux/performance.json'); graph=read('cuda-graph/summary.json')
rows=[]
for name in ('refund_1','refund_4','support_28','long_context_4'):
 c=p['cases'][name]['summary']; rows.append('| '+name+' | '+' | '.join(f"{c[v]['median_ms']:.1f}"for v in ('auto','math_only','expanded_kv'))+' |')
krows=['| '+n+f" | {v['count']:,} | {v['duration_ms']:.2f} | {100*v['duration_ms']/summary['kernel_duration_sum_ms']:.1f}% |"for n,v in sorted(groups.items(),key=lambda x:-x[1]['duration_ms'])]
brows=['| '+n+f" | {v['median_ms']:.1f} | {v['min_ms']:.1f}–{v['max_ms']:.1f} |"for n,v in p['cases']['support_28']['summary'].items()]
grows=['| '+mode+f" | {graph['summary'][mode]['wall_ms']:.1f} | {min(x['wall_ms']for x in samples):.1f}–{max(x['wall_ms']for x in samples):.1f} | {graph['summary'][mode]['cuda_event_ms']:.1f} |"for mode,samples in graph['samples'].items()]
sha=hashlib.sha256((ROOT/'src/subset/engine.py').read_bytes()).hexdigest()
assert sha=='9e417f108b896c9de7c92e96fe4b9784e765c8a817bcf3665cf54ec84bca7096'
report='''# Engine profiling: numerical stability and GPU optimization

Completed 18 September 2026. This supplements REPORT.md with controlled numerical checks, Linux backend experiments, CUDA graph replay, and direct GPU kernel activity. All experiments live in this dedicated Inference Engine project. Production engine.py is unchanged; no commits or pushes were made.

## Findings and recommended order

1. Temperature is not causing random answers. The engine scores fixed answer tokens with softmax; it does not sample generated tokens. Repeated inputs with different random seeds produced identical probabilities and did not consume CUDA RNG state. Changing temperature rescales the same scores deterministically.
2. Build a reusable CUDA graph path for common input shapes first, retaining an eager fallback. A fixed-shape single-question core experiment was about 4.7x faster by median, but this is not an end-to-end engine speed claim. Shape buckets, reusable buffers, cache lifecycle and bounded graph memory still need implementation and validation.
3. Use a runtime with working fused attention and verify actual dispatch. Linux automatically used FlashAttention for prefix processing and memory-efficient attention for padded branches. This is already optimized attention; forcing more FlashAttention everywhere is not automatically beneficial.
4. Group similar-length branches and remove unnecessary cache cloning. These are smaller, concrete engine changes supported by the measurements below.
5. Target fusion of normalization, rotary-position operations and gated activations if launch overhead remains after graph replay. The trace shows many elementwise kernels; SiLU and multiply are visible candidates. First try compiler/library fusion, then a small custom kernel where profiling supports it. Do not start by replacing vendor matrix multiplication kernels.

These improvements are not additive, and probability parity must be checked independently from speed. Training is not required for these runtime optimizations.

## Environment and method

RTX 3050 Laptop GPU, 4 GB VRAM, SM 8.6; driver 566.07. Model Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, FP16 weights. PyTorch 2.6.0+cu124 and Transformers 5.5.4. Numerical diagnostics used Windows Python 3.11.9. Linux experiments used Ubuntu through WSL, Python 3.12.3, an isolated user-cache virtual environment and a local copy of the same snapshot. No global Linux packages or Windows runtime dependencies were changed.

Performance sweep: two warmups per variant, nine rounds with rotated variant order; timings completed before collecting profiler traces. Single-question input had 193 tokens; refund branches 152–195; support branches 359–448; long-context branches 1,093–1,136. Results are specific to this GPU, model and workload. Laptop scheduling/power noise is visible, especially for short requests. Cross-platform runs are not a controlled Windows-versus-Linux speed comparison.

## Numerical investigation

The full 28-question request repeated with seeds 0, 11 and 97 had exactly zero probability spread within either default or expanded-KV execution. CUDA RNG state was unchanged. Repeated temperature-0.5 runs were also identical; their probabilities matched the deterministic transformation of temperature-1 probabilities within 3.08e-8. The Linux sweep likewise had zero within-variant probability spread across nine runs.

The old absolute probability tolerance was 0.005 (0.5 percentage points). Exceeding it means two implementations disagree more than that; it does not establish which implementation is better or that either has lost task accuracy.

For four selected disagreement cases, we computed an FP32-arithmetic reference on the SAME FP16-rounded weights, with mathematical attention and TF32 disabled. This is neither the original full-precision checkpoint nor labeled ground truth. Layers were temporarily promoted one at a time to fit memory, then restored; the original FP16 outputs matched exactly afterward. Independent versus cached/batched FP32 calculations agreed on all four classes and differed by at most 0.0000115 in probability.

| Question | Default FP16 maximum absolute error vs FP32 | Expanded-KV error vs FP32 |
|---|---:|---:|
| security_incident | 0.028994 | 0.021321 |
| primary_department | 0.003697 | 0.002143 |
| postmortem_required | 0.003143 | 0.001823 |
| contract_penalty_applicable | 0.009681 | 0.004964 |

The alternative was closer on these four selected cases; this is not proof of general superiority. For security_incident, yes probabilities were 73.20%, 72.43%, and 70.30% for default, alternative, and reference. For contract_penalty_applicable, the FP32 yes probability was 49.916%, while both FP16 paths were slightly above 50%: numerical precision can change a decision near a tie.

Next validation should track absolute probability drift, changed labels, and closeness to decision thresholds, alongside accuracy/calibration on labeled examples. Keep the existing tolerance visible; do not simply loosen it to make a speed change pass. PyTorch documents that mathematically equivalent batched and fused computations need not be bitwise identical: https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html

## End-to-end Linux attention measurements

Median full-request latency in milliseconds. Auto is the existing engine under Linux SDPA dispatch. Math-only disables fused SDPA. Expanded-KV explicitly expands grouped-query KV heads; it is an experimental variant, not shared zero-copy KV storage.

| Workload | Auto | Math-only | Expanded KV |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Long-context auto was about 40% lower latency than math-only. This comparison demonstrates the benefit of fused attention as a whole, not FlashAttention alone: the multi-question traces contain both Flash and efficient attention. Expanded KV provided no consistent further Linux improvement. The earlier Windows expanded-KV benefit should not be generalized to this runtime. The single-question expanded-KV timing is especially noisy and is not evidence of a reliable improvement.

Actual operator dispatch: single question, 28 Flash calls; refund/long four-question requests, 28 Flash plus 28 efficient calls; support 28-question request, 28 Flash plus 112 efficient calls. Direct CUPTI additionally confirmed Flash and CUTLASS memory-efficient native kernels for the four-question request. SDPA backend behavior: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html

## Branch batching and cache experiments

Same 28-question support workload; milliseconds. Min/max are observed sample ranges, not confidence intervals.

| Variant | Median | Observed range |
|---|---:|---:|
'''+ '\n'.join(brows)+'''

Sorting by rendered token length before making batches of eight reduced median latency about 17.6%. Fields remain mapped by question ID. This changes batching and padding, not the rubric or semantic questions. Maximum probability drift versus auto was 0.007639, with all 28 selected labels unchanged.

The shallow-cache experiment cloned DynamicCache/layer metadata while allowing the existing repeat/append operations to allocate their own tensors. It removed redundant prefix tensor deep-copying; it did not eliminate per-branch KV duplication. Median latency was about 3.6% lower, with exactly identical probabilities and unchanged peak allocated memory. This small gain needs further confirmation when integrated.

Batch 16 was modestly faster, but batch 28 was much slower. More parallel branches do not guarantee lower latency on a 4 GB GPU. Peak PyTorch allocation rose from roughly 3,135 MiB (auto) to 3,561 MiB (batch 28), excluding some driver/runtime memory. All support variants selected the same 28 labels, but several changed probabilities by more than the old tolerance.

## CUDA graph experiment

Separate fixed-shape 152-token refund_requested question, five warmups and 20 alternating eager/replay pairs. Input tensors and selected output-head weights were prepared before measurement. The measured core includes backbone and answer scoring, but excludes tokenization, input copies, head-row preparation and JSON. It is not directly comparable to the 193-token single-question end-to-end case above.

| Core execution | Median wall ms | Observed wall range ms | Median CUDA event ms |
|---|---:|---:|---:|
'''+ '\n'.join(grows)+'''

Median core speedup was about 4.7x. This supports prioritizing reduced CPU dispatch/launch overhead, including for a single question. CUDA event elapsed time includes gaps while the CPU submits work; it is not the sum of active kernel execution. The experiment used about 2,970 MiB of total PyTorch allocated memory after capture, not that much additional graph-only memory.

Graph versus eager maximum probability difference was 0.002454. A same-length changed-input test verified the replay reads updated inputs, with difference 0.001103 versus fresh eager execution. Neither was bitwise identical, and broad equivalence is not established. The exact numerical cause was not isolated. Production integration needs more shapes, padding/masks, independent and branched requests, cache isolation, bounded memory and end-to-end checks. Graph reuse mechanism: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

## Native GPU activity

PyTorch/Kineto exported CPU operators and CUDA API calls but omitted GPU kernel records in this environment. Those traces are marked cuda_api_only; missing kernel durations are null rather than zero. A separate direct CUPTI collector successfully captured 2,625 native kernel records for a warmed four-question request, with zero dropped records and no collector errors.

| Heuristic kernel group | Calls | Summed GPU duration ms | Share of kernel duration |
|---|---:|---:|---:|
'''+ '\n'.join(krows)+f'''

Summed kernel duration: {summary['kernel_duration_sum_ms']:.2f} ms; union of recorded kernel intervals: {summary['kernel_interval_union_ms']:.2f} ms. First-to-last native kernel span: {summary['kernel_span_ms']:.2f} ms. Instrumented request wall time: {summary['profiled_wall_ms']:.2f} ms; separately measured CUDA-event elapsed time: {summary['profiled_cuda_event_ms']:.2f} ms. These are separate clock/measurement scopes and must not be forced into an exact additive breakdown. Instrumented timing is not the unprofiled benchmark median.

The trace has substantial gaps between kernels, consistent with the graph experiment and CPU launch costs. The gaps are not all proven CPU overhead: this collector does not measure occupancy, memory bandwidth, hardware counters, or all competing GPU activity. Kernel groups are name-based, not exact model-layer attribution. Matrix multiplication dominates active kernel time, while 1,304 elementwise launches make fusion worth investigating. Attention is already a small part of active kernel time on this short workload; long inputs can behave differently.

## Artifacts and reproducibility

- deep-windows-numerical/numerical.json: seeds, RNG-state checks, temperature check, FP32 reference and restoration checks.
- deep-linux/performance.json: all raw timing samples, probability outputs, parity, memory and dispatch summaries.
- deep-linux/*-native-trace.json: CPU/API timelines; GPU kernel events absent.
- cuda-graph/summary.json: scope, every paired timing, numerical and changed-input checks.
- cupti-model/summary.json: raw native kernel records and real engine answers.
- cupti-model/kernel-analysis.json: grouped and per-kernel duration totals.
- cupti-model/gpu-trace.json: GPU-only Chrome trace with relative CUPTI timestamps; not aligned to the separate CPU trace.
- scripts/profile_deep.py, profile_cuda_graph.py and profile_cupti.py: diagnostic experiment harnesses.
- scripts/setup_linux_profiler.sh and run_linux_graph.sh: isolated Linux setup/launcher.
- scripts/summarize_deep_profile.py: generates this report and derived native artifacts without rerunning inference.

Production engine.py SHA256 verified unchanged: {sha}.

No new model training, production custom kernels, production graph integration, or new comparison against the external HF replica was performed. The next implementation should start with graph reuse and measured batching/cache improvements, then rerun these same workload and numerical checks before claiming a production speedup.
'''
(OUT/'DEEP_REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps({'report':str(OUT/'DEEP_REPORT.md'),'native_groups':dict(groups),'engine_sha256':sha},indent=2))