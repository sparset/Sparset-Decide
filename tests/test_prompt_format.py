"""Input/rubric boundaries and parity with the evaluated prompt candidate."""
import argparse,json,sys,unittest
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from subset.engine import DecisionEngine
from subset.schema import Choice,Noul,Score
from subset.__main__ import add_model_arguments
from test_decision_engine import TinyTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prompt_bias_variants import install,select

class PromptFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(91)
        cfg=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096)
        cfg._attn_implementation="sdpa";cls.model=Qwen2ForCausalLM(cfg).eval()

    def test_default_matches_evaluated_candidate_for_dynamic_inputs(self):
        current=DecisionEngine(self.model,TinyTokenizer())
        reference=DecisionEngine(self.model,TinyTokenizer(),prompt_format="readable",answer_encoding="letters")
        install(reference);select(reference,"delimited")
        questions={"route":Choice("Choose a department.",{"billing":"Payments","shipping":"Delivery"}),"refund":Noul("Refund requested?"),"level":Score("Rate the impact.",["Low","High"])}
        for context in ('A duplicate charge appeared. Please refund it.', 'Quoted: "Question:"\n<input_text> is a literal tag in the request.'):
            self.assertEqual(current.prepare(context,questions),reference.prepare(context,questions))
            self.assertEqual(current.decide(context,questions)["answers"],reference.decide(context,questions)["answers"])

    def test_boundaries_keep_input_and_rubric_separate_without_changing_options(self):
        class Capture(TinyTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages;return super().apply_chat_template(messages,**kwargs)
        tokenizer=Capture();engine=DecisionEngine(self.model,tokenizer)
        context='Line one\nA quoted "instruction".'
        engine.prepare(context,{"x":Choice("Which category?",{"name":"Meaning","other":"Alternative"})})
        text=tokenizer.messages[-1]["content"]
        self.assertTrue(text.startswith('<input_text>\n'+json.dumps(context)+'\n</input_text>'))
        self.assertIn('Evaluation instructions (not input text):\nWhich category?',text)
        self.assertIn('A. "name": "Meaning"',text)
        self.assertTrue(text.endswith('Return only the letter of the best option.'))

    def test_defaults_and_legacy_formats_remain_explicit(self):
        parser=argparse.ArgumentParser();add_model_arguments(parser)
        args=parser.parse_args([])
        self.assertEqual((args.prompt_format,args.answer_encoding),("delimited","letters"))
        for value in ("readable","json"):
            self.assertEqual(parser.parse_args(["--prompt-format",value]).prompt_format,value)
        with self.assertRaises(ValueError):DecisionEngine(self.model,TinyTokenizer(),prompt_format="unknown")
