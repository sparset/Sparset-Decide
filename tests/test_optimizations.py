"""Regression checks for replay input freshness, cache isolation and fused kernels."""
import unittest
import torch
from test_decision_engine import TinyTokenizer
from subset.engine import DecisionEngine
from subset.schema import Noul, Choice
from transformers import Qwen2Config,Qwen2ForCausalLM


def tiny(device='cpu',**kwargs):
    torch.manual_seed(71)
    cfg=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(cfg).eval().to(device)
    return DecisionEngine(model,TinyTokenizer(),branch_batch_size=2,**kwargs)

class CacheTests(unittest.TestCase):
    def test_cpu_graph_request_uses_eager_fallback(self):
        e=tiny(cuda_graphs=True)
        result=e.decide('A customer request',{'a':Noul('A refund?')})
        self.assertEqual(e._graphs.stats['captures'],0)
        self.assertEqual(e._graphs.stats['fallbacks'],1)
        self.assertIn('a',result['answers'])

    def test_head_cache_invalidation(self):
        e=tiny();h=torch.randn(1,32)
        with torch.inference_mode():
            before=e._project(h,[2]);e.head.weight[e.answer_ids[0]].add_(5*h[0])
            e.clear_optimization_cache();after=e._project(h,[2])
        self.assertGreater(float(after[0,0]),float(before[0,0]))

    def test_fork_does_not_modify_prefix(self):
        e=tiny();rows=e.prepare('Earlier customer context',{'a':Noul('Requested?')})
        with torch.inference_mode():
            _,cache=e._hidden(rows,use_cache=True)
            original=[(x.keys.clone(),x.values.clone())for x in cache.layers]
            branch=e._fork_cache(cache);branch.batch_repeat_interleave(2)
            e._hidden([[4,5],[5,6]],cache=branch,prefix_length=len(rows[0]),use_cache=True)
        for layer,(k,v)in zip(cache.layers,original):
            torch.testing.assert_close(layer.keys,k,rtol=0,atol=0);torch.testing.assert_close(layer.values,v,rtol=0,atol=0)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GraphTests(unittest.TestCase):
    def test_changed_inputs_lengths_and_cache_replay(self):
        e=tiny('cuda',cuda_graphs=True,max_graphs=4)
        q={'a':Noul('Refund?'),'b':Choice('Where should this request go?',{'x':'Billing','y':'Sales','z':'Support'}),'c':Noul('An outage?')}
        try:
            for context in ('Customer asked for a refund.','Customer asked for a return.','A different shorter message.'):
                e.cuda_graphs=False;expected=e.decide(context,q)
                e.cuda_graphs=True;e.decide(context,q);actual=e.decide(context,q)
                for k in q:
                    for label,value in expected['answers'][k]['probabilities'].items():
                        self.assertAlmostEqual(value,actual['answers'][k]['probabilities'][label],places=5)
            self.assertGreater(e._graphs.stats['replays'],0)
            self.assertEqual(e._graphs.stats['fallbacks'],0,e._graphs.last_fallback)
        finally:e.clear_optimization_cache()

    def test_same_shape_different_padding_refreshes_masks_and_indices(self):
        e=tiny('cuda',cuda_graphs=True)
        try:
            with torch.inference_mode():
                for rows in ([[3,4,5],[6]], [[8],[9,10,11]], [[6,2],[9,5,8]]):
                    expected,_=e._hidden_eager(rows);actual,_=e._hidden(rows)
                    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            self.assertEqual(e._graphs.stats['captures'],1)
            self.assertEqual(e._graphs.stats['replays'],3)
        finally:e.clear_optimization_cache()

    def test_capture_failure_falls_back_and_is_not_retried(self):
        e=tiny('cuda',cuda_graphs=True)
        def fail(*args,**kwargs):raise RuntimeError('injected unsupported capture')
        e._graphs._capture=fail
        q={'a':Noul('Refund?')}
        try:
            first=e.decide('Customer request',q);second=e.decide('Customer request',q)
            self.assertEqual(first['answers'],second['answers'])
            self.assertEqual(e._graphs.stats['captures'],0)
            self.assertEqual(len(e._graphs.failed),1)
        finally:e.clear_optimization_cache()

    def test_eviction_and_changed_same_shape_inputs(self):
        e=tiny('cuda',cuda_graphs=True,max_graphs=1)
        try:
            with torch.inference_mode():
                for rows in ([[3,4,5]],[[6,7,8]],[[4,8]],[[6,2,9]]):
                    expected,_=e._hidden_eager(rows);actual,_=e._hidden(rows)
                    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            self.assertEqual(len(e._graphs.entries),1)
            self.assertGreater(e._graphs.stats['evictions'],0)
        finally:e.clear_optimization_cache()

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:from subset import kernels
        except ImportError as exc:raise unittest.SkipTest(str(exc))
        cls.k=kernels

    def test_rmsnorm_and_swiglu(self):
        with torch.inference_mode():
            for dtype in (torch.float16,torch.bfloat16,torch.float32):
                for rows in (1,31,257):
                    x=torch.randn(rows,1536,device='cuda',dtype=dtype);w=torch.randn(1536,device='cuda',dtype=dtype)
                    ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(dtype)*w
                    torch.testing.assert_close(self.k.rmsnorm(x,w,1e-6),ref,atol=.016 if dtype==torch.bfloat16 else .002,rtol=.008 if dtype==torch.bfloat16 else .002)
                    g=torch.randn(rows,8960,device='cuda',dtype=dtype);u=torch.randn_like(g)
                    torch.testing.assert_close(self.k.swiglu(g,u),torch.nn.functional.silu(g)*u,atol=.016 if dtype==torch.bfloat16 else .002,rtol=.008 if dtype==torch.bfloat16 else .002)

    def test_model_scoped_fusion_with_graphs_and_changed_context(self):
        e=tiny('cuda',cuda_graphs=True,fused_kernels=('rmsnorm','swiglu','rope'))
        questions={'a':Noul('Was a refund requested?'),'b':Noul('Was shipping mentioned?')}
        try:
            for context in ('A customer requests a refund.', 'A customer requests a return.'):
                e.cuda_graphs=False;e.fused_kernels=();reference=e.decide(context,questions)
                e.fused_kernels=('rmsnorm','swiglu','rope');e.cuda_graphs=True
                e.decide(context,questions);actual=e.decide(context,questions)
                for key in questions:
                    for label,p in reference['answers'][key]['probabilities'].items():
                        self.assertAlmostEqual(p,actual['answers'][key]['probabilities'][label],places=5)
                self.assertFalse(e._fusion.active)
            self.assertEqual(e._graphs.stats['fallbacks'],0,e._graphs.last_fallback)
        finally:e.clear_optimization_cache()

    def test_rope_noncontiguous_queries_and_broadcast_positions(self):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        with torch.inference_mode():
            for batch in (1,3):
                for cos_batch in (1,batch):
                    q=torch.randn(batch,17,12,128,device='cuda',dtype=torch.float16).transpose(1,2)
                    k=torch.randn(batch,17,2,128,device='cuda',dtype=torch.float16).transpose(1,2)
                    phase=torch.randn(cos_batch,17,128,device='cuda',dtype=torch.float16);c,s=phase.cos(),phase.sin()
                    ref=apply_rotary_pos_emb(q,k,c,s);actual=self.k.rope(q,k,c,s)
                    for a,b in zip(actual,ref):torch.testing.assert_close(a,b,atol=0,rtol=0)

if __name__=='__main__':unittest.main()