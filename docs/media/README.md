# JSON response comparison video

[Watch or download the MP4](subset-comparison.mp4). Silent, 11 seconds, 1920 × 1080, 60 fps.

The request is **“Implement a Java LRU cache.”** The task is to route it to `coding`, `reasoning`, or `fast`; the expected choice is `coding`.

This is a deliberately successful demonstration, not an accuracy estimate. Case `routing_031` was chosen from existing results before fresh latency measurements, not selected for fastest latency.

All three methods use Qwen/Qwen2.5-1.5B-Instruct in FP16 on an RTX 3050 Laptop GPU (4 GB). They ran separately, with three warmups and five timed requests each. The video shows each method's actual median trial, aligned at elapsed zero. Loading and warmups are excluded.

| Method | Median latency | Output |
|---|---:|---|
| Subset | 336.97 ms | Correct route, valid schema |
| harshatheg/Qwen-2.5-1B-RLCD, PyTorch/CUDA implementation | 573.36 ms | Correct route, valid schema |
| Standard Qwen, streamed JSON generation | 5,201.72 ms | Coding inside an extra wrapper; invalid schema |

These are separate recordings from the [250-case benchmark](../benchmarks.md). The laptop was discharging on battery at the post-run power check. Streaming callbacks also add overhead to the standard Qwen recording, so its time is not a non-streaming benchmark result.

The replay runs at 1× speed after a two-second introduction. Base-model text appears at its recorded token-arrival times. Subset and the HF implementation display assembled JSON when their requests finish, using the comparison adapter's common answer representation. Probabilities and base output text are unchanged. Frame timing introduces at most 16.67 ms of rounding.

Subset used CUDA graphs off and RMSNorm, SwiGLU, and RoPE fusions on. Capture code: [capture_json_video_trace_v2.py](../../scripts/capture_json_video_trace_v2.py).

- Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- [HF implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) revision: `2af86848be75847ccb3553b0941cc51d6ef7e4e9`
- Source trace SHA-256: `b1cd0801491d8476f1f25879a03322ad11b7265c9c629154316e3cf59c6d9ea0`
