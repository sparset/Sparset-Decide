"""Run dynamic decision requests with a reusable workflow and local model."""

import argparse
import json
import sys
from pathlib import Path

from .schema import question_from_dict
from .workflow import Workflow, read_json


def load_request(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("questions"), dict):
        raise ValueError("Input needs context and a questions object")
    return data, {key: question_from_dict(value) for key, value in data["questions"].items()}


def add_model_arguments(parser):
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", help="Pin an HF commit for reproducible runs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--prompt-format", choices=["delimited", "readable", "json"], default="delimited", help="Delimited input/rubric separation; readable and json reproduce earlier prompts")
    parser.add_argument("--answer-encoding", choices=["labels", "letters"], default="letters", help="letters uses fast one-token IDs; labels scores complete literal labels")
    parser.add_argument("--max-label-tokens", type=int, default=64)
    parser.add_argument("--attention", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--branch-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true", help="Capture/reuse exact-shape CUDA graphs; reuse one engine instance for warm performance")
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument("--graph-budget-mb", type=int, default=192)
    parser.add_argument("--no-efficient-cache", action="store_true")
    parser.add_argument("--fused-kernels", nargs="*", choices=["rmsnorm", "swiglu", "rope", "residual_norm"], default=[], help="Optional Linux/Triton Qwen2 inference kernels")

    parser.add_argument("--shared-attention", choices=["auto", "off", "on"], default="auto",
                        help="Shared physical prefix storage, with automatic compatibility/shape fallback")
    parser.add_argument("--short-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length-grouping", action=argparse.BooleanOptionalAction, default=True)


def build_engine(args):
    from .engine import DecisionEngine

    return DecisionEngine.from_pretrained(
        args.model, device=args.device, dtype=args.dtype, attention=args.attention,
        revision=args.revision, local_files_only=args.local_files_only,
        branch_batch_size=args.branch_batch_size, max_input_tokens=args.max_input_tokens,
        efficient_cache=not args.no_efficient_cache, cuda_graphs=args.cuda_graphs,
        max_graphs=args.max_graphs, graph_budget_mb=args.graph_budget_mb,
        fused_kernels=args.fused_kernels, shared_attention=args.shared_attention,
        specialize_short=args.short_attention, length_aware=args.length_grouping, prompt_format=args.prompt_format, answer_encoding=args.answer_encoding, max_label_tokens=args.max_label_tokens,
    )


def write_result(result, output):
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def interactive(engine, workflow, *, mode="parallel", output=None):
    print("Model ready. Enter context on multiple lines; finish with /run. /quit exits.")
    print("Each request starts fresh. Prompt entries apply only to that request.")
    while True:
        try:
            print("\nContext:")
            lines = []
            while True:
                line = input()
                if line == "/quit":
                    return
                if line == "/run":
                    break
                lines.append(line)
            prompts = {}
            for key in workflow.question_names:
                default = workflow.default_prompt(key)
                label = f"Prompt for {key}"
                if default:
                    label += f" [Enter keeps: {default}]"
                value = input(label + ": ")
                if value == "/quit":
                    return
                if value.strip() or not default:
                    prompts[key] = value
            result = workflow.run(engine, context="\n".join(lines), prompts=prompts, mode=mode)
            write_result(result, output)
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return
        except ValueError as exc:
            print(f"Invalid request: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, help="Reusable rubric (default: workflow.json)")
    parser.add_argument("--input", type=Path, help="Legacy combined context/questions JSON")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--request", type=Path, help="Per-call JSON: context/messages and prompt/prompts")
    source.add_argument("--context", help="Context text for this request")
    source.add_argument("--context-file", type=Path, help="UTF-8 context text file")
    source.add_argument("--interactive", action="store_true", help="Keep the model loaded and enter repeated requests")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=None, help="Initialize runtime before requests (default: on for interactive sessions)")
    parser.add_argument("--prompt", help="Question instruction override for a single-question workflow")
    parser.add_argument("--output", type=Path, help="Save JSON (interactive mode overwrites with the latest result)")
    parser.add_argument("--mode", default="parallel", choices=["parallel", "independent"])
    add_model_arguments(parser)
    args = parser.parse_args()
    dynamic = (args.request is not None or args.context is not None or
               args.context_file is not None or args.interactive or args.prompt is not None)
    if args.input is not None and (args.workflow is not None or dynamic):
        parser.error("--input is a legacy combined request; use --workflow with dynamic inputs")
    if args.prompt is not None and (args.request is not None or args.interactive):
        parser.error("Supply the prompt inside --request or at the interactive prompt")
    try:
        if args.input is not None:
            data, questions = load_request(args.input)
            context = data["context"]
            if not isinstance(context, str):
                raise ValueError("context must be a string")
        else:
            data = read_json(args.workflow or Path("workflow.json"))
            workflow = Workflow(data)
            if not args.interactive:
                if args.request is not None:
                    request = read_json(args.request)
                elif dynamic:
                    context = (args.context_file.read_text(encoding="utf-8-sig")
                               if args.context_file is not None else args.context)
                    request = {"context": context, "prompt": args.prompt}
                elif "context" in data:
                    # Retain support for existing combined workflow files.
                    request = {"context": data["context"]}
                else:
                    raise ValueError("Supply --context, --context-file, --request, or --interactive")
                context, questions = workflow.bind_request(request)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Loading {args.model} on {args.device}. Please wait...", file=sys.stderr, flush=True)
    engine = build_engine(args)
    if args.warmup is True or (args.warmup is None and args.interactive):
        print("Warming up the model before accepting requests...", file=sys.stderr, flush=True)
        engine.warmup()
    if args.interactive:
        print(f"Workflow: {args.workflow or 'workflow.json'}; questions: {', '.join(workflow.question_names)}", flush=True)
        interactive(engine, workflow, mode=args.mode, output=args.output)
    else:
        write_result(engine.decide(context, questions, mode=args.mode), args.output)


if __name__ == "__main__":
    main()
