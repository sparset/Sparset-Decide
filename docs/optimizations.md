# Optional inference optimizations

The model remains a dense Qwen2/Qwen3/Llama language model. These options change execution, not training or workflow schema. The rubric remains in workflow.json.

## Use

For repeated requests, keep one DecisionEngine instance alive:

```python
from subset.engine import DecisionEngine

engine = DecisionEngine.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",
    attention="sdpa",
    cuda_graphs=True,
    fused_kernels=("rmsnorm", "swiglu", "rope"),
)
# Call engine.decide(context, questions) repeatedly with application-supplied questions.
```

For the CLI:

```bash
subset --device cuda --cuda-graphs --fused-kernels rmsnorm swiglu rope --workflow workflow.json --interactive
```

A one-shot CLI invocation pays graph capture and kernel compilation overhead without reusing the graph on later requests. Warm latency measurements apply to a persistent engine, not process startup. Omit --cuda-graphs for one-shot use. Each fusion can be enabled separately. Omit --fused-kernels to use standard PyTorch kernels.

## Requirements

Cache optimizations use the existing inference dependencies and fall back to deep copying for unsupported cache layouts. CUDA graphs require a compatible CUDA PyTorch build and SDPA; CPU and unsupported graph shapes use ordinary execution. Qwen2.5-1.5B on an RTX 3050 4 GB with PyTorch 2.6.0+cu124 is the full-model validation target; graph correctness tests also use a tiny real Qwen2 model. Qwen3 and Llama remain supported by the ordinary engine, but their graph paths have not been benchmarked here.

Custom kernels currently require Linux/WSL, CUDA, Triton 3.2.0, Transformers 5.5.4 and dense Qwen2 with SiLU activation. Install the fusion extra with an appropriate CUDA PyTorch build. Triton also needs a working C compiler and Python development headers for its launch helper. Unsupported model/version combinations fail explicitly when fusion is requested; these adapters do not silently patch a different architecture. The current Windows environment can use the ordinary engine without installing Triton.

The local benchmark environment is under ~/.cache/subset/profiling-20260918. scripts/run_optimization.sh sets its isolated compiler and launches benchmarks. This launcher is a convenience for this workstation, not a portable installer.

## What is implemented

- CUDA graphs: record the backbone execution and reuse it for the exact batch size, token width, prefix length and mask mode. Token IDs, positions, final-token indices, masks and prefix-cache values are refreshed on every replay. Graph storage is bounded; default maximum is four graphs with a 192 MiB retained-allocation budget and conservative admission based on prefix KV size. Shapes that cannot fit or cannot capture fall back to eager execution. The budget is not a cap on total model memory or peak capture memory. No approximate input truncation or padding bucket is introduced.
- Cache handling: copy dense DynamicCache layer metadata rather than redundantly deep-copying prefix tensors, then use the existing operations that allocate branch tensors. Physical per-branch KV duplication still occurs. The small FP32 answer-head row selection is also cached. This does not cache answers or retain a semantic answer across requests.
- RMSNorm: one Triton kernel computes normalization and rescaling.
- SwiGLU: one Triton kernel performs SiLU activation followed by multiplication.
- RoPE: one Triton launch rotates query and key position representations, including strided query/key inputs. The adapters are local to the model instance, not global changes to Transformers.

Fusion retains the reference operation's intermediate low-precision rounding where possible. Outputs are not promised to be bitwise identical: reduction order and other backend choices can still change probabilities.

## Lifetime and concurrency

Calls to decide on engines sharing the same optimized model are serialized to protect graph buffers and model-scoped fusion selection. Separate model instances have separate locks. This is not a multi-stream concurrent serving scheduler.

Weights, dtype and device must remain fixed while serving. After changing weights in place, call engine.clear_optimization_cache() before the next request. Construct a new engine after moving or recasting the model. clear_optimization_cache clears selected head rows, graphs and remembered failed shapes. Per-engine metadata reports graph captures, replays, fallbacks, evictions, resident entries and the last fallback reason; counters are cumulative over that engine's lifetime.

Graph buffers and some internal graph outputs retain input-derived data until reused, evicted or cleared. They are private to the engine and overwritten before replay. Clear the optimization cache when explicitly discarding this retained state. No cross-request context KV cache is reused without refreshing it.

Cache optimizations are enabled by default; --no-efficient-cache disables them for comparison. CUDA graphs and custom kernels remain opt-in because startup cost, shape variability and available VRAM can make them slower for some workloads.

## Verification and benchmarks

Run scripts/run_optimization_tests.py from the configured environment. It covers the existing tiny-model semantics plus cache isolation, head-cache invalidation, graph replay with changed inputs, fallback, eviction, and custom-kernel comparisons across dimensions and numeric types.

scripts/benchmark_optimizations.py runs each stage against the saved original engine in outputs/decision-engine/optimization-20260918/baseline_engine.py. Raw results include full probability outputs, paired timing samples, cold timings, changed-input comparisons and memory statistics. That original snapshot is an ignored local benchmark artifact, not a runtime dependency. Use a persistent immutable reference implementation when reproducing on another checkout.

A probability shift of 0.005 is 0.5 percentage points; it is not a measured 0.5% loss in accuracy. The benchmark reports label agreement separately. The four-field refund example is the only small labeled smoke case in this suite; it is insufficient to estimate sub-percent accuracy changes. Larger verified evaluation datasets are needed for that.

Measured implementation results are saved in outputs/decision-engine/optimization-20260918/REPORT.md. Earlier exploratory timings are retained separately and should not be substituted for these end-to-end implementation measurements.
## Measured configuration guidance

On the measured RTX 3050 workloads, the final graph-plus-fusion configuration was
1.78x faster for one question and 1.69x for four short questions after warmup.
For larger workloads, all three fused kernels without graphs gave useful gains
with lower memory overhead; graph reuse added little to the 28-question case.
One near-50% decision changed with the combined path, so the results do not
establish a sub-percent accuracy guarantee. The saved report includes that case.

Graphs currently require matching exact shapes, not approximate length buckets.
New shapes incur capture cost; highly variable input lengths can defeat reuse.
The measured warm speedups should not be applied to one-shot or always-new-shape
traffic. Use ordinary execution with fusion for such traffic, or benchmark graph
reuse on the actual input distribution before enabling it.

## Branch execution update

See [automatic branch optimizations](branch-optimizations.md) for shared-prefix
attention, short-query dispatch, length grouping, and optional residual/normalization
fusion. Captures now reuse a model-scoped stream; repeated new shapes no longer
create a fresh cuBLAS stream workspace on every capture. Historical timings above
remain historical and should not replace the new paired benchmark.
