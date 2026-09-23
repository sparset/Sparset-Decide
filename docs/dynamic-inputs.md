# Dynamic requests and reusable workflows

The workflow stores answer types, options, and scoring criteria. Its `instructions`
are optional default questions. Each call supplies fresh context or conversation
history and may replace the question instructions. This does not change model
weights, candidate scoring, parallel branching, or optimization settings.

## Try it interactively

From the project directory in PowerShell, using the installed local environment:

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.cache/decision-engine/huggingface'
.\.cache\decision-engine\venv\Scripts\python.exe -m sparset-decide --interactive --device cuda --local-files-only
```

The model loads once, then runs a synthetic initialization call before displaying
`Model ready`. This moves first-use runtime overhead into startup; it does not
cache answers or train the model. `--no-warmup` skips it. Paste context across multiple lines, type `/run` on a line
by itself, then enter a prompt for each configured question. Press Enter to use
that question's saved default. Type `/quit` to exit. There is no automatic history:
each new request starts fresh. Paste the complete history you want evaluated.
`--output outputs/result.json` saves the latest result, overwriting the prior one.

For the routing rubric (which deliberately has no fixed question):

```powershell
.\.cache\decision-engine\venv\Scripts\python.exe -m sparset-decide --workflow examples/decision_engine/router.workflow.json --interactive --device cuda --local-files-only
```

Use a prompt such as `Which model should handle the next response?`. The example
fast/reasoning/coding labels describe hypothetical capabilities; replace these
with your own model names and descriptions. This selects a label, not a downstream
model invocation, and the sample is not a validated routing policy.

## One request from the command line

```powershell
.\.cache\decision-engine\venv\Scripts\python.exe -m sparset-decide --workflow examples/decision_engine/router.workflow.json --prompt "Which model should answer this?" --context "Write and debug a Python CSV parser." --device cuda --local-files-only
```

Use `--context-file conversation.txt` for a multiline UTF-8 text file. Each CLI
invocation reloads the model; use interactive mode or the Python API for reuse.

A request JSON can contain `context`, `prompt`, `prompts`, and `messages` only:

```json
{
  "prompt": "Which model should handle the next response?",
  "context": "Prefer the least expensive model capable of the task.",
  "messages": [
    {"role": "user", "content": "Please fix this Python traceback..."}
  ]
}
```

Pass it with `--workflow examples/decision_engine/router.workflow.json --request
examples/decision_engine/router.request.json`. The request supplies new input;
no context is inherited from the workflow. `prompt` applies to a single-question
workflow. With multiple questions, use `prompts` keyed by their workflow names:

```json
{
  "context": "I was charged twice. Please refund the extra payment.",
  "prompts": {
    "department": "Which team should investigate this message?",
    "refund_requested": "Is a refund explicitly requested?"
  }
}
```

Unspecified questions use their saved instructions. Missing required prompts,
unknown question names, and invalid schemas fail before loading the model.
Overrides replace instructions rather than append to them; keep stable grading
criteria in `criteria`. All configured questions still run. The engine does not
infer extra branches from free-form prose.

## Use in an application, with one resident model

```python
from sparset_decide import Workflow
from sparset_decide.engine import DecisionEngine

# Once at application startup. Use your existing engine settings here.
engine = DecisionEngine.from_pretrained(device="cuda", local_files_only=True)
engine.warmup()  # Once at startup, before accepting requests.
workflow = Workflow.load("examples/decision_engine/router.workflow.json")

def route(conversation, context="", prompt="Which model should handle the next response?"):
    result = workflow.run(
        engine,
        prompt=prompt,
        context=context,
        messages=conversation,
    )
    return result["answers"]["model"]

answer = route([
    {"role": "user", "content": "Please implement a Python CSV parser."}
])
print(answer["choice"], answer["probabilities"])
```

`messages` accepts ordered text-only `{role, content}` objects, with roles system,
developer, user, assistant, or tool. They are serialized as context data for the
routing decision, not executed as the engine's chat instructions. No multimodal
content, tool-call structures, automatic history storage, or truncation is added.
Serialization is not a security boundary against prompt injection. The normal
input-token limit applies to context, history, question, and rubric together.

The API calls the same `engine.decide` path. Keep optimization flags appropriate
to your platform; dynamic inputs may change shapes and require new graph captures.
The engine refreshes request data, and no previous answer or conversation is reused.
No network server, OpenAI endpoint, or Ollama dependency is involved.

## Existing requests

`--input examples/decision_engine/refund.json` still accepts the original combined
context/questions format. A pre-existing combined `workflow.json` still runs
without extra flags. The supplied default `workflow.json` now contains only the
reusable refund rubric, so supply dynamic inputs or choose `--interactive`.
The lower-level `engine.decide(context, questions)` API remains unchanged.

## Prompt format and timing

Choice and Score present option names and descriptions, JSON-escaped to preserve
boundaries, and score their A/B/C IDs. Noul still uses false/true. The default is
`--answer-encoding letters`; literal labels are available explicitly with
`--answer-encoding labels`. `--prompt-format delimited` is the default and separates input text from evaluation
instructions automatically. `--prompt-format readable` preserves the earlier
layout; `--prompt-format json` selects the older JSON catalog. Neither changes
the answer encoding. See [literal-label scoring](literal-labels.md) for multi-token
labels and latency implications. Changing answer encoding changes the model task;
it can change probabilities substantially and does not guarantee better accuracy.

`--cuda-graphs` remains opt-in. It can reduce repeated matching-shape latency on
Windows too, but the first new shape still pays capture cost. Startup warmup avoids
capturing a synthetic graph and does not pre-capture unknown user request shapes.
One-shot runs skip warmup by default; `--warmup` enables it explicitly. Loading,
initialization, and cold graph capture are separate from steady inference latency.
