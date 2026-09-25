# Parallel decision engine: experimental baseline

This engine works with an existing instruction model. Fine-tuning is not required
for prefix caching, batched branches, or direct option scoring. Training can
improve task accuracy and answer-format compliance; it must be evaluated separately.
The default is **Qwen/Qwen2.5-1.5B-Instruct**, the same parent checkpoint used by
the Hugging Face replica. The replica's published Apple Silicon results use a
4-bit MLX conversion; this baseline uses ordinary PyTorch weights on CUDA or CPU.

## What is implemented

1. Accept dynamic Choice, Noul, and Score questions with English descriptions.
   The default delimiter prompt separates input text from the evaluation rubric;
   see [prompt format and validation](prompt-format.md).
2. Score boolean (Noul) questions with `false` and `true`. Choice and Score use
   single-token A/B/C IDs by default. Optional literal-label scoring evaluates
   actual labels completely, including termination for overlapping prefixes.
3. Render each complete chat input, then find its exact common **token** prefix.
4. Run that prefix through the transformer backbone once, without vocabulary logits.
5. Repeat its cache into a batch of independent question suffixes. Right padding,
   position IDs, and attention masks preserve branch isolation.
6. For single-token candidates, project final decision states onto the required
   LM-head rows in FP32. Apply masked softmax and construct typed results in
   Python. No text is generated. Multi-token labels use sequence likelihoods.
7. Single-token candidates need one decision position. Multi-token candidates
   reuse each question prompt and evaluate known continuations in batches. See
   [literal-label scoring](literal-labels.md) for the additional cost and limitations.

The ordinary backend physically repeats cache storage per active branch. The
[automatic shared-attention path](branch-optimizations.md) avoids this duplication
for eligible batches. Questions exceeding
`--branch-batch-size` run in bounded sequential chunks, each sharing the prefill.
The ordinary path is not paged zero-copy storage; the specialized path uses a
custom kernel that reads a shared prefix and private suffixes.
SDPA uses the backend supported by the installed PyTorch, dtype, GPU and masks;
requesting SDPA does not prove that a FlashAttention kernel was selected.

## Supported scope

- Dense Qwen2, Qwen3, and Llama-style causal backbones with a normal Linear LM head.
  Qwen2 is covered by numerical parity tests; other accepted families require
  their own real-checkpoint verification before production use.
- One device per engine; CUDA or CPU. No model offloading, quantization wrapper,
  diffusion, custom scoring head, or multi-GPU scheduling yet.
- At most 26 options/question, 32 questions/request by default, and 2048 tokens
  per complete question plus candidate continuation by default. Labels are bounded
  to 64 tokens. Overlength input fails instead of truncating.
- Noul returns P(true) as `noul`, preserving the public `no`/`yes` probability keys.
  Multi-token boolean encodings use sequence scoring in the optional labels mode.
  Score returns the mean of zero-indexed rubric levels.
- Probabilities are normalized likelihoods of answer tokens within the allowed set.
  They are **not calibrated correctness estimates**. No arbitrary confidence
  floor is applied. Prompt wording and option order can affect predictions.
- Fine-tuned checkpoints can use the same engine if they retain the supported
  architecture, tokenizer contract, and output head. A saved LoRA adapter must
  first be loaded/merged into its matching base model; it is not a standalone model.

## Installation

Use a separate environment. Install a CUDA-enabled PyTorch build compatible with
your GPU driver from https://pytorch.org/get-started/locally/ before the package.
The `inference` extra alone does not guarantee a CUDA-enabled wheel.

```powershell
python -m venv .cache/decision-engine/venv
.cache/decision-engine/venv/Scripts/python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.cache/decision-engine/venv/Scripts/python.exe -m pip install -e '.[inference]'
```

The CUDA 12.4 line is a reference Windows setup for this workspace's NVIDIA
driver, not a universal hardware prescription. Use a suitable wheel elsewhere.
On CPU, install `.[inference]` in a normal environment and select `--device cpu`.
The first checkpoint load downloads roughly 3 GB of weights. Keep model and
dependency caches on a disk with adequate space. No training or paid API is used.

## Run

The default rubric file is `workflow.json` in your current working directory.
Supply new context and prompts with `--interactive`, `--context`, `--context-file`,
or `--request`. See [dynamic inputs and application integration](dynamic-inputs.md).
`--input` accepts a legacy combined request; it remains required for the benchmark
command so benchmark inputs are explicit.

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.cache/decision-engine/huggingface'
.cache/decision-engine/venv/Scripts/python.exe -m subset --input examples/decision_engine/refund.json --device cuda --output outputs/decision-engine/refund.json
```

Use `--revision <HF-commit>` to pin a checkpoint, `--local-files-only` after
downloading, and `--branch-batch-size 1` to reduce peak branch-cache memory.
`--mode independent` recomputes every complete question separately, using the
identical prompt and scoring contract. `--attention eager` is a diagnostic fallback.

Programmatic API:

```python
from subset import Choice, Noul
from subset.engine import DecisionEngine

engine = DecisionEngine.from_pretrained(device="cuda")
result = engine.decide(
    "I was charged twice. Please refund the duplicate charge.",
    {
        "department": Choice("Which department handles this?", {
            "billing": "Payment errors and duplicate charges",
            "shipping": "Parcel delivery problems",
        }),
        "refund_requested": Noul("Does the customer ask for a refund?"),
    },
)
print(result["answers"])
```

## Test and benchmark

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p test_decision_engine.py -v
.cache/decision-engine/venv/Scripts/python.exe -m subset.benchmark --input examples/decision_engine/refund.json --device cuda --repeats 5 --include-json --output outputs/decision-engine/benchmark.json
```

The offline tests use a tiny randomly initialized real Qwen2 transformer. They
verify cached/uncached numerical equivalence, ragged padding, cache isolation,
chunking, option masking, selected-row projection, and single-question execution.
They establish implementation correctness, not useful language understanding.

The benchmark alternates mode order, synchronizes GPU timings, records raw
outputs and versions, and reports paired probability/decision agreement. Model
loading and warmup are excluded; tokenization is included. The optional ordinary
JSON baseline emits labels only (an easier output contract than distributions).
Its malformed or truncated results remain visible; timing alone is not success.
Labeled fields are scored only when `expected_labels` is supplied. The included
refund example is a smoke test, not a general evaluation dataset.

### Replica workload import

```powershell
python scripts/import_decision_replica_presets.py
.cache/decision-engine/venv/Scripts/python.exe -m subset.benchmark --input outputs/decision-engine/replica-presets/support_triage.request.json --device cuda --repeats 5 --output outputs/decision-engine/support-triage.json
```

The importer pins the public repository revision, records original file hashes,
retains the source bytes, and converts supported fields. These are unlabeled
latency scenarios. The 255-option case is explicitly unsupported, never silently
reduced. Some source monetary values are malformed; they remain unchanged.
Current presets contain more fields than some published benchmark rows.

This runner does **not** execute the original HF engine. Comparing our paired
baseline is not proof of beating that engine. A direct competition requires both
engines on the same hardware, precision, input revision, warmup and repeated timing
protocol, plus verified semantic accuracy. M4 Max/MLX numbers are reference only.

### First local CUDA check (2026-09-17)

The pinned checkpoint `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` ran on an RTX
3050 4 GB Laptop GPU, using FP16 backbone weights, FP32 selected-head scoring,
PyTorch 2.6.0+cu124 and Transformers 5.5.4. The four-field refund smoke test
produced matching parallel/independent decisions in all 20 comparisons. Maximum
probability difference was 0.002318, below the unchanged 0.005 tolerance.

Both modes answered **3/4 labeled fields correctly**: they missed the explicit
refund request. This is a concrete accuracy limitation, not a production-ready
result. Their median times were 699 ms parallel and 1475 ms independent (2.11x).
Parallel timings varied from 386 to 2572 ms; treat these as noisy local smoke
measurements, not a stable performance guarantee or a win against the HF engine.

Raw results: `outputs/decision-engine/qwen2_5_cuda_refund_fp32_head.json`.
The earlier FP16-head run is retained as `qwen2_5_cuda_refund.json`; it failed
the numerical threshold (0.005828 difference). Changing the small scoring step
to FP32 fixed this particular rounding problem without relaxing the threshold.
The model itself remains unmodified and its probabilities remain uncalibrated.

Three imported 28-field workloads were also measured, with one warmup and three
paired repetitions each, using the same loaded model and batch size 8:

| Workload | Parallel median | Independent median | Ratio | Largest probability difference |
| --- | ---: | ---: | ---: | ---: |
| Support triage | 1696 ms | 6296 ms | 3.71x | 0.009744 |
| Fintech fraud | 1625 ms | 5871 ms | 3.61x | 0.007427 |
| Code security | 1685 ms | 5939 ms | 3.52x | 0.011394 |

Every selected label matched between modes (252/252 comparisons), but **all
three workloads failed the 0.005 probability-parity gate**. FP32 head scoring
does not eliminate differences from the FP16 backbone. Characterizing these
differences against a higher-precision reference remains necessary before
claiming probability equivalence. These scenarios have no verified labels, so
agreement is not accuracy. Support triage included a 16.76-second parallel
outlier; all samples are retained in the reports. No original HF engine was run.

Reports are `outputs/decision-engine/{support_triage,fintech_fraud,code_security}_cuda.json`.
The exact sequential suite runner is saved at
`outputs/decision-engine/run_replica_suite.py`, and installed versions at
`outputs/decision-engine/environment.txt`.

Source: https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD
Source files declare Apache-2.0. The engine implementation here is independently
written; only the explicit import command downloads their scenario data.

## Next optimization gates

Profile prefill, suffix evaluation, projection, and CPU/GPU synchronization before
adding specialized kernels. Then evaluate shared paged cache storage, compilation,
fused probability operations, and supported quantization. Preserve numerical
parity where intended; remeasure task accuracy and calibration after precision or
training changes. A stable measured baseline comes before speed claims.

## Natural boolean scoring comparison

Boolean prompts request `true` or `false` directly, without A/B remapping. A shared
format-neutral system instruction preserves prefix sharing with Choice and Score
questions. Mixed batches project a small union of token rows, then gather each
question's permitted candidates. Head caches are keyed by the actual token IDs.

The response metadata identifies each field's `answer_encoding`. The top-level
`probability_status` is now `uncalibrated_answer_token_probabilities`; consumers
matching the old status string must update. Question definitions and answer
structures are unchanged.

Run `scripts/compare_natural_boolean.py` in the configured Linux/CUDA environment
to compare the new engine, the saved pre-change engine, ordinary JSON generation,
and the pinned HF replica. Its local source snapshots and results are in
`outputs/decision-engine/natural-boolean-20260918`. These ignored artifacts must
be supplied when reproducing on another checkout. The comparison changes both
the prompt and candidate tokens; it does not isolate their individual effects.

Current answer encoding and probability status are described in [literal-label scoring](literal-labels.md). Historical benchmark sections above describe the versions tested at those dates.
