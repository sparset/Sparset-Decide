"""Compare complete candidate scoring with independent full-model likelihoods."""
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from subset.engine import DecisionEngine
from subset.schema import Choice, Noul, Score
from test_decision_engine import TinyTokenizer

class LiteralLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(71)
        config=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096)
        config._attn_implementation="sdpa"
        cls.model=Qwen2ForCausalLM(config).eval()
        cls.engine=DecisionEngine(cls.model,TinyTokenizer(),answer_encoding="labels",branch_batch_size=2)

    @torch.inference_mode()
    def reference(self,q,context="Please refund the charge."):
        prompt=self.engine.prepare(context,{"x":q})[0]
        scores=[]
        for tokens in self.engine._candidate_tokens(q):
            ids=torch.tensor([prompt+tokens[:-1]])
            logits=self.model(ids).logits[0,len(prompt)-1:].float().log_softmax(-1)
            scores.append(logits.gather(1,torch.tensor(tokens)[:,None]).sum())
        return torch.stack(scores).softmax(0)

    def test_complete_labels_match_model_and_independent(self):
        for labels in ({"red":"a","blue":"b","green":"c"},{"billing support":"a","billing dispute":"b"},{"red":"a","red fox":"b"}):
            q=Choice("Pick the best category",labels)
            result=self.engine.decide("Please refund the charge.",{"x":q})
            actual=torch.tensor(list(result["answers"]["x"]["probabilities"].values()))
            torch.testing.assert_close(actual,self.reference(q),atol=2e-6,rtol=2e-5)
            independent=self.engine.decide("Please refund the charge.",{"x":q},mode="independent")
            torch.testing.assert_close(actual,torch.tensor(list(independent["answers"]["x"]["probabilities"].values())),atol=2e-6,rtol=2e-5)

    def test_mixed_fields_and_single_token_projection(self):
        qs={"long":Choice("category",{"red fox":"a","blue whale":"b"}),"bool":Noul("Refund requested?"),"score":Score("Severity",["low","high"])}
        result=self.engine.decide("Please refund the charge.",qs)
        self.assertEqual(list(result["answers"]),list(qs))
        for key,q in qs.items():
            actual=torch.tensor(list(result["answers"][key]["probabilities"].values()))
            torch.testing.assert_close(actual,self.reference(q),atol=2e-6,rtol=2e-5)

    def test_validation_and_prefix_termination(self):
        q=Choice("category",{"red":"a","red fox":"b"})
        self.assertTrue(all(x[-1]==1 for x in self.engine._candidate_tokens(q)))
        limited=DecisionEngine(self.model,TinyTokenizer(),answer_encoding="labels",max_label_tokens=2)
        with self.assertRaisesRegex(ValueError,"max_label_tokens"):
            limited.decide("context",{"x":q})
        with self.assertRaisesRegex(ValueError,"distinct complete"):
            self.engine._candidate_tokens(Choice("category",{"a":"a",chr(ord("a")+180):"b"}))

    def test_prompt_contains_actual_labels_without_letter_ids(self):
        class Capture(TinyTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages
                return super().apply_chat_template(messages,**kwargs)
        tokenizer=Capture();e=DecisionEngine(self.model,tokenizer,answer_encoding="labels")
        e.prepare("context",{"x":Choice("category",{"billing":"Payment","shipping":"Parcel"})})
        text=tokenizer.messages[-1]["content"]
        self.assertIn('"billing": "Payment"',text)
        self.assertNotIn("A =",text)
        self.assertIn("exact option label",text)

    def test_default_cli_encoding_and_removed_correction(self):
        import argparse, contextlib, io
        from subset.__main__ import add_model_arguments
        parser=argparse.ArgumentParser();add_model_arguments(parser)
        self.assertEqual(parser.parse_args([]).answer_encoding,"letters")
        self.assertEqual(parser.parse_args(["--answer-encoding","letters"]).answer_encoding,"letters")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--answer-bias","contextual"])
        with self.assertRaises(TypeError):
            DecisionEngine(self.model,TinyTokenizer(),answer_bias="contextual")

if __name__=="__main__": unittest.main()
