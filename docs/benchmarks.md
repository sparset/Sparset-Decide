# Local 250-case development benchmark

These are local measurements of Sparset Decide, the pinned Hugging Face implementation, and ordinary Qwen JSON generation. They do not compare against Jev itself or establish general model accuracy.

## Results

| Configuration | Correct + schema-valid | Schema-valid | Median request latency |
|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct, unconstrained JSON generation | 23/250 (9.2%) | 35/250 (14%) | 4,358.842 ms |
| Qwen-2.5-1B-RLCD, PyTorch/CUDA | 187/250 (74.8%) | 250/250 (100%) | 408.484 ms |
| Sparset Decide, CUDA graphs on | 225/250 (90.0%) | 250/250 (100%) | 950.695 ms |
| Sparset Decide, CUDA graphs off | 225/250 (90.0%) | 250/250 (100%) | 167.107 ms |

Ratios of median latencies: 4,358.842 / 167.107 = **26.1×** versus standard JSON generation; 408.484 / 167.107 = **2.4×** versus the HF implementation. These are not means of per-case speedups.

## Setup

- Hardware: NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM, WSL/Linux.
- Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP16 backbone. Sparset Decide scores the selected output-head rows in FP32.
- Comparison implementation: [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), revision `2af86848be75847ccb3553b0941cc51d6ef7e4e9`, using its PyTorch/CUDA path and the same Qwen checkpoint. Its repository name says 1B; the tested checkpoint is 1.5B. Its published MLX/Apple Silicon benchmarks are separate.
- Runtime: PyTorch 2.6.0+cu124, Transformers 5.5.4, Triton 3.2.0.
- Sparset Decide settings: delimited prompts, letter option IDs, efficient cache enabled, RMSNorm/SwiGLU/RoPE fusions enabled, SDPA attention, branch batch size 8. CUDA graphs disabled for the headline result. The README's basic installation does not enable optional fusions automatically.
- Requests: identical inputs and case order; each method uses its own prompting and output procedure. The comparison measures the full configured systems, not the isolated effect of any one kernel.
- Timing: one timed request per case, in sequential method blocks, with model loading excluded. Request preparation, inference, result processing, and any new graph captures are included. These are not cold-start results.

## What the percentages mean

**Correct + schema-valid** means the selected answer matches the expected label and the full requested JSON passes validation. Each answer must contain the choice and a probability for every allowed option; probabilities must be finite, in range, sum to one within rounding tolerance, and agree with the selected choice.

The ordinary Qwen baseline generates that JSON without constrained decoding or repair. Its 9.2% therefore measures structured-task success, not standalone classification accuracy. Sparset Decide and the HF implementation assemble their JSON from model scores. The observed 100% schema validity is separate from answer correctness or probability calibration.

## Dataset and limitations

The suite contains 100 BANKING77 messages from ten intents and 150 constructed routing, boolean, and severity cases. Some cases were used during prompt development, so this is a development set rather than an untouched evaluation set.

Every case requests one decision. This suite does not measure the scaling benefit of multiple parallel fields. Repeated timing trials, controlled laptop power conditions, and a held-out dataset are needed to establish broader performance and quality conclusions.

Disabling CUDA graphs was a diagnostic follow-up to the initial graph-enabled run. All 250 choices stayed the same; six cases had probability differences greater than one percentage point, with a maximum difference of 4.42 points. Probabilities remain uncalibrated token scores.

## Evidence and scripts

Measurements were captured on 18 September 2026. The local evidence folder is `outputs/decision-engine/matched-latency-250-20260918/`, containing `requests.jsonl`, `no-graphs-requests.jsonl`, environment records, source snapshots, and verification results. Raw artifacts and downloaded models are excluded from Git; this page is a summary of the saved results.

Relevant scripts: [graph-enabled comparison](../scripts/benchmark_matched_250.py), [graphs-disabled comparison](../scripts/benchmark_matched_250_no_graphs.py), and [report generation](../scripts/report_matched_250.py). These historical scripts require the local frozen dataset and upstream source snapshot described in their paths; a fresh clone alone is not a complete benchmark reproduction bundle.
