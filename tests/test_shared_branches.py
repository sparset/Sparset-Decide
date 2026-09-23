import unittest
import importlib.util
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from sparset_decide.engine import DecisionEngine
from sparset_decide.schema import Choice,Noul
from test_decision_engine import TinyTokenizer

class LengthTests(unittest.TestCase):
    def test_grouping_restores_output_order_and_cpu_fallback(self):
        torch.manual_seed(2);torch.set_num_threads(2)
        cfg=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,attention_dropout=0.)
        cfg._attn_implementation='sdpa'
        model=Qwen2ForCausalLM(cfg).eval()
        e=DecisionEngine(model,TinyTokenizer(),branch_batch_size=2,length_aware=True,shared_attention='auto')
        q={str(i):Noul('Was it requested?'+(' Background detail.'*20 if i%2 else ''))for i in range(8)}
        result=e.decide('Customer context.',q);e.length_aware=False;ref=e.decide('Customer context.',q)
        self.assertEqual(list(result['answers']),list(q))
        self.assertTrue(result['metadata']['optimizations']['reordered'])
        self.assertEqual(result['metadata']['optimizations']['shared_branch_batches'],0)
        for k in q:
            for label,p in ref['answers'][k]['probabilities'].items():self.assertAlmostEqual(p,result['answers'][k]['probabilities'][label],places=6)

@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('triton') is not None, 'CUDA and Triton required')
class SharedTests(unittest.TestCase):
    def test_capture_stream_reused_across_shapes_and_engines(self):
        from test_optimizations import tiny
        import gc
        e=tiny('cuda',cuda_graphs=True,max_graphs=1)
        other=DecisionEngine(e.model,e.tokenizer,cuda_graphs=True,max_graphs=1)
        allocations=[]
        try:
            with torch.inference_mode():
                for engine,width in ((e,3),(other,4),(e,5),(other,6),(e,7),(other,8)):
                    e.clear_optimization_cache();other.clear_optimization_cache();gc.collect()
                    engine._hidden([list(range(3,3+width))]);torch.cuda.synchronize()
                    if len(allocations)==0:stream=e._fusion.capture_stream
                    self.assertIs(e._fusion.capture_stream,stream)
                    engine.clear_optimization_cache();gc.collect();torch.cuda.synchronize()
                    allocations.append(torch.cuda.memory_allocated())
            self.assertLess(max(allocations[1:])-min(allocations[1:]),2*1024**2,allocations)
        finally:e.clear_optimization_cache();other.clear_optimization_cache()

    def test_attention_against_float32_reference_ragged_gqa_and_short_tiles(self):
        from sparset_decide.shared_attention import attention
        import torch.nn.functional as F
        torch.manual_seed(9)
        with torch.inference_mode():
            for dtype in (torch.float16,torch.bfloat16):
                for d,t,p in ((32,1,37),(64,17,129),(128,65,251)):
                    b,h,hk=2,4,2
                    q=torch.randn(b,h,t,d,device='cuda',dtype=dtype)
                    k=torch.randn(b,t,hk,d,device='cuda',dtype=dtype).transpose(1,2)
                    v=torch.randn_like(k)
                    pk=torch.randn(1,p,hk,d,device='cuda',dtype=dtype).transpose(1,2)
                    pv=torch.randn_like(pk)
                    lengths=torch.tensor([t,max(1,t//3)],device='cuda')
                    keys=torch.cat((pk.expand(b,-1,-1,-1),k),2).repeat_interleave(h//hk,1).float()
                    vals=torch.cat((pv.expand(b,-1,-1,-1),v),2).repeat_interleave(h//hk,1).float()
                    j=torch.arange(p+t,device='cuda');i=torch.arange(t,device='cuda')+p
                    mask=(j[None,None,:]<=i[None,:,None])&(j[None,None,:]<(p+lengths)[:,None,None])
                    ref=F.scaled_dot_product_attention(q.float(),keys,vals,attn_mask=mask[:,None])
                    for short in (False,True):
                        actual=attention(q,k,v,pk,pv,lengths,d**-.5,short)
                        torch.testing.assert_close(actual.float(),ref,atol=.012 if dtype==torch.bfloat16 else .003,rtol=.03)

    def test_shared_model_probabilities_prefix_immutable_and_changed_inputs(self):
        torch.manual_seed(10)
        cfg=Qwen2Config(vocab_size=256,hidden_size=128,intermediate_size=256,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,attention_dropout=0.)
        cfg._attn_implementation='sdpa';model=Qwen2ForCausalLM(cfg).eval().half().cuda()
        e=DecisionEngine(model,TinyTokenizer(),shared_attention='on',specialize_short=True,
                         fused_kernels=('rmsnorm','swiglu','rope','residual_norm'),branch_batch_size=2)
        q={'a':Noul('Refund?'),'b':Choice('Which route?',{'x':'Billing','y':'Sales'}),'c':Noul('Human requested?')}
        for context in ('Please refund my payment.','Do not refund my payment.'):
            e.shared_attention='off';ref=e.decide(context,q)
            e.shared_attention='on';actual=e.decide(context,q)
            self.assertEqual(actual['metadata']['optimizations']['shared_branch_batches'],2)
            self.assertIsNone(e._fusion.branch_state)
            for key in q:
                for label,p in ref['answers'][key]['probabilities'].items():
                    self.assertLess(abs(p-actual['answers'][key]['probabilities'][label]),.003)
        with torch.inference_mode():
            _,cache=e._hidden([[4,5,6,7]],use_cache=True)
            saved=[(x.keys.clone(),x.values.clone())for x in cache.layers]
            e._hidden_shared([[8,9],[10]],cache,4)
            for layer,(k,v)in zip(cache.layers,saved):
                torch.testing.assert_close(k,layer.keys,atol=0,rtol=0);torch.testing.assert_close(v,layer.values,atol=0,rtol=0)

    def test_compiler_resource_failure_falls_back_without_retry(self):
        from test_optimizations import tiny
        from triton.runtime.errors import OutOfResources
        e=tiny('cuda',shared_attention='on')
        q={'a':Noul('Refund?'),'b':Noul('Human?')}
        e.shared_attention='off';reference=e.decide('Customer request.',q)
        e.shared_attention='on'
        e._can_share=lambda *args:e._shared_available is not False
        attempts=[]
        def failure(*args):
            attempts.append(1)
            raise OutOfResources(100,1,'shared memory')
        e._hidden_shared=failure
        for _ in range(2):
            actual=e.decide('Customer request.',q)
            self.assertEqual(actual['answers'],reference['answers'])
            self.assertEqual(actual['metadata']['optimizations']['shared_branch_batches'],0)
        self.assertEqual(len(attempts),1)
        self.assertIsNone(e._fusion.branch_state)

    def test_fused_residual_norm_preserves_rounded_sum(self):
        from sparset_decide.kernels import residual_rmsnorm
        with torch.inference_mode():
            for dtype in (torch.float16,torch.bfloat16,torch.float32):
                x=torch.randn(13,1536,device='cuda',dtype=dtype);r=torch.randn_like(x);w=torch.randn(1536,device='cuda',dtype=dtype)
                added,actual=residual_rmsnorm(x,r,w,1e-6);ref_sum=x+r
                ref=w*(ref_sum.float()*torch.rsqrt(ref_sum.float().square().mean(-1,keepdim=True)+1e-6)).to(dtype)
                torch.testing.assert_close(added,ref_sum,atol=0,rtol=0)
                torch.testing.assert_close(actual,ref,atol=.02 if dtype==torch.bfloat16 else .003,rtol=.01)
