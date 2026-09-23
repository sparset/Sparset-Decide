"""Correctness checks using a tiny real transformer; no model downloads needed."""

import unittest

from sparset_decide.schema import Choice, Noul, Score, question_from_dict


class SchemaTests(unittest.TestCase):
    def test_primitives_and_score_expectation(self):
        self.assertEqual(Choice("route", {"a": "alpha", "b": "beta"}).answer([0.2, 0.8])["choice"], "b")
        self.assertAlmostEqual(Noul("true?").answer([0.15, 0.85])["noul"], 0.85)
        self.assertAlmostEqual(Score("severity", ["minor", "moderate", "severe"]).answer([0.05, 0.2, 0.75])["score"], 1.7)

    def test_schema_rejects_unknown_fields_and_invalid_vectors(self):
        with self.assertRaises(ValueError):
            question_from_dict({"type": "noul", "instructions": "yes?", "criterai": []})
        with self.assertRaises(ValueError):
            Choice("route", {"a": ""})
        for p in ([0.1, 0.1], [float("nan"), 0.5], [-0.1, 1.1]):
            with self.assertRaises(ValueError):
                Noul("yes?").answer(p)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        if text in ("false", "true"):
            return [2 if text == "false" else 3]
        return [10 + ord(c) - 65 if "A" <= c <= "Z" else 40 + ord(c) % 180 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("return_dict") is False
        return self.encode("\n".join(message["content"] for message in messages) + "\nAssistant:\n")


class TransformerParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest(f"Install the inference extra for transformer tests: {exc}")
        from sparset_decide.engine import DecisionEngine

        torch.set_num_threads(2)
        torch.manual_seed(19)
        cls.torch = torch
        config = Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=4096,
                             attention_dropout=0.0, use_sliding_window=False)
        config._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(config).eval()
        cls.engine = DecisionEngine(cls.model, TinyTokenizer(), answer_encoding="letters", max_input_tokens=2048,
                                    branch_batch_size=2)
        cls.context = "A duplicate charge appeared. A refund was requested."
        cls.questions = {
            "department": Choice("Which department handles this?", {"billing": "Payment errors", "sales": "New purchases", "shipping": "Parcel delivery"}),
            "refund": Noul("Was a refund explicitly requested?"),
            "severity": Score("How severe is this?", ["Informational", "One customer affected", "Everyone affected", "Service destroyed"]),
        }

    def assertParity(self, questions, batch_size):
        self.engine.branch_batch_size = batch_size
        a = self.engine.decide(self.context, questions)
        b = self.engine.decide(self.context, questions, mode="independent")
        for key in questions:
            pa = a["answers"][key]["probabilities"]
            pb = b["answers"][key]["probabilities"]
            self.assertEqual(pa.keys(), pb.keys())
            for label in pa:
                self.assertAlmostEqual(pa[label], pb[label], places=6)
        return a

    def test_shared_prefix_and_ragged_suffixes_match_full_independent_forward(self):
        a = self.assertParity(self.questions, 8)
        self.assertGreater(a["metadata"]["shared_prefix_tokens"], 0)
        self.assertEqual(a["metadata"]["forward_calls"], 2)

    def test_chunked_cache_isolation_and_question_reordering(self):
        a = self.assertParity(self.questions, 2)
        b = self.assertParity(dict(reversed(list(self.questions.items()))), 1)
        self.assertEqual(a["metadata"]["forward_calls"], 3)
        for key in self.questions:
            for label, value in a["answers"][key]["probabilities"].items():
                self.assertAlmostEqual(value, b["answers"][key]["probabilities"][label], places=6)

    def test_single_question_uses_one_forward(self):
        a = self.assertParity({"refund": self.questions["refund"]}, 8)
        self.assertEqual(a["metadata"]["forward_calls"], 1)
        self.assertEqual(a["metadata"]["shared_prefix_tokens"], 0)

    def test_selected_projection_matches_full_vocabulary_logits(self):
        questions = {"refund": self.questions["refund"]}
        sequences = self.engine.prepare(self.context, questions)
        torch = self.torch
        with torch.inference_mode():
            hidden, _ = self.engine._hidden(sequences)
            selected = self.engine._project(hidden, [2], [True])
            full = self.model(input_ids=torch.tensor(sequences), use_cache=False).logits[:, -1]
            reference = full[:, list(self.engine._boolean_token_ids)].softmax(-1)
        torch.testing.assert_close(selected, reference, atol=1e-6, rtol=1e-6)

    def test_mixed_answer_tokens_match_full_head_and_do_not_alias_cache(self):
        torch = self.torch
        with torch.inference_mode():
            hidden = torch.randn(3, 32)
            self.engine._project(hidden, [2, 2, 2])  # Populate same-size letter cache first.
            actual = self.engine._project(hidden, [2, 3, 2], [True, False, True])
            full = self.engine.head(hidden).float()
            for i, ids in enumerate((self.engine._boolean_token_ids,
                                     self.engine._option_token_ids[:3], self.engine._boolean_token_ids)):
                reference = full[i, list(ids)].softmax(-1)
                torch.testing.assert_close(actual[i, :len(ids)], reference, atol=1e-6, rtol=1e-6)
                self.assertTrue(torch.equal(actual[i, len(ids):], torch.zeros_like(actual[i, len(ids):])))
            letters = self.engine._project(hidden[:1], [2])
            torch.testing.assert_close(letters[0], full[0, list(self.engine._option_token_ids[:2])].softmax(-1))

    def test_boolean_token_mapping_and_unsupported_tokenizer(self):
        from sparset_decide.engine import DecisionEngine
        q = {"refund": Noul("Refund requested?")}
        result = self.engine.decide(self.context, q)
        self.assertEqual(result["metadata"]["answer_encoding"], {"refund": "false_true"})
        self.assertEqual(list(result["answers"]["refund"]["probabilities"]), ["no", "yes"])
        self.assertEqual(result["answers"]["refund"]["noul"], result["answers"]["refund"]["probabilities"]["yes"])
        class SplitBooleanTokenizer(TinyTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [2, 3] if text in ("false", "true") else super().encode(text, add_special_tokens)
        e = DecisionEngine(self.model, SplitBooleanTokenizer(), answer_encoding="letters")
        with self.assertRaisesRegex(ValueError, "single-token"):
            e.prepare(self.context, q)
        e.prepare(self.context, {"route": Choice("Route?", {"a": "A", "b": "B"})})

    def test_overlength_requests_fail_without_truncating(self):
        with self.assertRaisesRegex(ValueError, "No truncation"):
            self.engine.decide("x" * 3000, self.questions)

    def test_readable_prompt_preserves_option_boundaries_and_boolean_format(self):
        from sparset_decide.engine import DecisionEngine
        class RecordingTokenizer(TinyTokenizer):
            messages = None
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return super().apply_chat_template(messages, **kwargs)
        tokenizer = RecordingTokenizer()
        engine = DecisionEngine(self.model, tokenizer, answer_encoding="letters", prompt_format="readable")
        engine.prepare("context", {"route": Choice("Choose", {"a\nb": 'description "one"', "c": "description two"})})
        text = tokenizer.messages[1]["content"]
        self.assertIn('A. "a\\nb": "description \\"one\\""', text)
        self.assertIn('B. "c": "description two"', text)
        original = DecisionEngine(self.model, tokenizer, prompt_format="json")
        q = {"bool": Noul("True?")}
        self.assertEqual(engine.prepare("context", q), original.prepare("context", q))

    def test_warmup_restores_graph_setting_without_recording_synthetic_graph(self):
        from sparset_decide.engine import DecisionEngine
        engine = DecisionEngine(self.model, TinyTokenizer(), cuda_graphs=True)
        metadata = engine.warmup()
        self.assertTrue(engine.cuda_graphs)
        self.assertEqual(len(engine._graphs.entries), 0)
        self.assertFalse(metadata["optimizations"]["cuda_graphs"])

    def test_dynamic_workflow_matches_direct_engine_with_changed_requests(self):
        from sparset_decide import Workflow
        workflow = Workflow({"questions": {"refund": {"type": "noul"}}})
        for context, prompt in ((self.context, "Refund requested?"),
                                ("New conversation without a refund", "Does this ask for a human?")):
            actual = workflow.run(self.engine, context=context, prompt=prompt)
            expected = self.engine.decide(context, {"refund": Noul(prompt)})
            self.assertEqual(actual["answers"], expected["answers"])

    def test_no_cache_leaks_between_requests(self):
        first = self.engine.decide(self.context, self.questions)
        self.engine.decide("Completely unrelated customer history", self.questions)
        second = self.engine.decide(self.context, self.questions)
        self.assertEqual(first["answers"], second["answers"])


if __name__ == "__main__":
    unittest.main()
