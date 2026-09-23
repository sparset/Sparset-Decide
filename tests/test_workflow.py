"""Request separation, history preservation, and resident-session regression checks."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sparset_decide import Workflow
from sparset_decide.__main__ import main


class RecordingEngine:
    def __init__(self):
        self.calls = []
        self.warmups = 0

    def warmup(self):
        self.warmups += 1

    def decide(self, context, questions, **kwargs):
        self.calls.append((context, questions, kwargs))
        return {"answers": {key: q.answer([1 / len(q.options)] * len(q.options))
                            for key, q in questions.items()}}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.data = {"questions": {"model": {"type": "choice", "criteria": {
            "fast": "Simple requests", "coding": "Programming tasks"}}}}

    def test_new_inputs_do_not_mutate_rubric_or_leak_between_calls(self):
        source = copy.deepcopy(self.data)
        workflow = Workflow(source)
        source["questions"]["model"]["criteria"]["fast"] = "Mutated caller data"
        engine = RecordingEngine()
        workflow.run(engine, context="First conversation", prompt="Route this")
        workflow.run(engine, context="Second conversation", prompt="Pick a model")
        self.assertEqual([c[0] for c in engine.calls], ["First conversation", "Second conversation"])
        self.assertEqual([c[1]["model"].instructions for c in engine.calls], ["Route this", "Pick a model"])
        self.assertEqual(dict(engine.calls[1][1]["model"].options), self.data["questions"]["model"]["criteria"])
        with self.assertRaisesRegex(ValueError, "Supply a prompt"):
            workflow.bind(context="No prior prompt should be retained")

    def test_history_roles_order_and_context_are_preserved_as_data(self):
        messages = [{"role": "system", "content": "Example conversation system text"},
                    {"role": "user", "content": "Hello \u263a"},
                    {"role": "assistant", "content": "Hello back"}]
        context, questions = Workflow(self.data).bind(prompt="Route?", context="Extra context", messages=messages)
        self.assertEqual(json.loads(context), {"context": "Extra context", "messages": messages})
        self.assertEqual(questions["model"].instructions, "Route?")

    def test_parallel_prompt_overrides_preserve_other_defaults(self):
        workflow = Workflow({"questions": {
            "refund": {"type": "noul", "instructions": "Refund requested?"},
            "severity": {"type": "score", "criteria": ["Minor", "Major"]}}})
        _, questions = workflow.bind(context="A new case", prompts={"severity": "How serious?"})
        self.assertEqual(questions["refund"].instructions, "Refund requested?")
        self.assertEqual(questions["severity"].instructions, "How serious?")
        with self.assertRaisesRegex(ValueError, "multiple questions"):
            workflow.bind(context="A case", prompt="Ambiguous")

    def test_local_only_loader_passes_cached_path_to_tokenizer_and_model(self):
        from sparset_decide.engine import DecisionEngine
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as folder:
            model = MagicMock()
            with patch("huggingface_hub.snapshot_download", return_value=folder) as snapshot, patch("transformers.AutoTokenizer.from_pretrained") as tokenizer, patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as loader, patch.object(DecisionEngine, "__init__", return_value=None):
                DecisionEngine.from_pretrained("Qwen/example", device="cpu", local_files_only=True, revision="pinned")
                snapshot.assert_called_once_with(repo_id="Qwen/example", revision="pinned", local_files_only=True)
                self.assertEqual(tokenizer.call_args.args[0], folder)
                self.assertEqual(loader.call_args.args[0], folder)
                snapshot.reset_mock()
                DecisionEngine.from_pretrained(folder, device="cpu", local_files_only=True)
                snapshot.assert_not_called()

    def test_bad_requests_fail_instead_of_ignoring_fields(self):
        workflow = Workflow(self.data)
        for request in (
            {"context": "x", "prompt": "p", "promt": "typo"},
            {"context": "x", "prompt": "p", "prompts": {}},
            {"context": "x", "prompts": {"typo": "p"}},
            {"context": "x", "prompt": ""},
            {"context": "", "prompt": "p"},
            {"prompt": "p", "messages": []},
            {"prompt": "p", "messages": [{"role": "user", "content": []}]},
            {"prompt": "p", "messages": [{"role": "bad", "content": "x"}]},
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                workflow.bind_request(request)

    def test_cli_request_and_legacy_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            workflow = folder / "workflow.json"
            request = folder / "request.json"
            legacy = folder / "legacy.json"
            workflow.write_text(json.dumps(self.data), encoding="utf-8")
            request.write_text(json.dumps({"context": "New input", "prompt": "Route?"}), encoding="utf-8")
            old = copy.deepcopy(self.data)
            old["context"] = "Legacy input"
            old["questions"]["model"]["instructions"] = "Legacy instruction"
            legacy.write_text(json.dumps(old), encoding="utf-8")
            for args, context in ((["--workflow", str(workflow), "--request", str(request)], "New input"),
                                  (["--workflow", str(workflow), "--context", "Inline input", "--prompt", "Route?"], "Inline input"),
                                  (["--input", str(legacy)], "Legacy input")):
                engine = RecordingEngine()
                with patch("sys.argv", ["sparset-decide", *args]), patch("sys.stdout", new_callable=io.StringIO), patch("sparset_decide.__main__.build_engine", return_value=engine) as loader:
                    main()
                loader.assert_called_once()
                self.assertEqual(engine.calls[0][0], context)

    def test_cli_bad_request_does_not_load_model_or_inherit_old_context(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workflow.json"
            data = copy.deepcopy(self.data)
            data["context"] = "Must not silently use this old context"
            path.write_text(json.dumps(data), encoding="utf-8")
            with patch("sys.argv", ["sparset-decide", "--workflow", str(path), "--prompt", "Route?"]), patch("sys.stderr", new_callable=io.StringIO), patch("sparset_decide.__main__.build_engine") as loader:
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)
                loader.assert_not_called()

    def test_interactive_loads_once_and_resets_request_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workflow.json"
            data = copy.deepcopy(self.data)
            data["questions"]["model"]["instructions"] = "Saved default"
            path.write_text(json.dumps(data), encoding="utf-8")
            engine = RecordingEngine()
            entries = ["First line", "Second line", "/run", "New instruction",
                       "Next context", "/run", "", "/quit"]
            with patch("sys.argv", ["sparset-decide", "--workflow", str(path), "--interactive"]), patch("builtins.input", side_effect=entries), patch("sys.stdout", new_callable=io.StringIO), patch("sparset_decide.__main__.build_engine", return_value=engine) as loader:
                main()
            loader.assert_called_once()
            self.assertEqual(engine.warmups, 1)
            self.assertEqual([c[0] for c in engine.calls], ["First line\nSecond line", "Next context"])
            self.assertEqual([c[1]["model"].instructions for c in engine.calls], ["New instruction", "Saved default"])


if __name__ == "__main__":
    unittest.main()
