"""Measure fused-kernel launch shapes in CUDA graphs to isolate GPU execution."""
import sys,os,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CC',str(Path.home()/'.cache/sparset-decide/profiling-20260918/tools/zig-cc'))
import torch,triton
from sparset_decide import kernels as k
OUT=ROOT/'outputs/decision-engine/optimization-20260918'
@torch.inference_mode()
def bench(fn):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(50):fn()
    samples=[]
    for _ in range(9):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/50)
    return {'median_us':statistics.median(samples),'samples_us':samples}
@torch.inference_mode()
def main():
    result={'scope':'GPU-only per-operation latency averaged over 50 operations in a CUDA graph, nine samples; excludes Python dispatch, allocations outside capture and full-model context. Used only for kernel launch-shape selection.','cases':{}}
    for rows in (193,424,1176):
        g=torch.randn(rows,8960,device='cuda',dtype=torch.float16);u=torch.randn_like(g);out=torch.empty_like(g);n=g.numel()
        refs=torch.nn.functional.silu(g)*u
        data={'pytorch':bench(lambda:torch.nn.functional.silu(g)*u)}
        for block in (256,512,1024,2048):
            fn=lambda:k._swiglu[(triton.cdiv(n,block),)](g,u,out,n,block,enable_fp_fusion=False)
            fn();diff=float((out-refs).abs().max());data[str(block)]={**bench(fn),'max_absolute_difference':diff}
        result['cases']['swiglu_'+str(rows)]=data
        print('swiglu',rows,{x:round(v['median_us'],3)for x,v in data.items()},flush=True)
        (OUT/'kernel-launch-sweep.json').write_text(json.dumps(result,indent=2)+'\n')
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    for batch,t in ((1,193),(4,106),(8,147)):
        q=torch.randn(batch,t,12,128,device='cuda',dtype=torch.float16).transpose(1,2)
        key=torch.randn(batch,t,2,128,device='cuda',dtype=torch.float16).transpose(1,2)
        phase=torch.randn(batch,t,128,device='cuda',dtype=torch.float16);cos,sin=phase.cos(),phase.sin()
        oq=torch.empty(q.shape,device='cuda',dtype=q.dtype);ok=torch.empty(key.shape,device='cuda',dtype=key.dtype)
        nq,nk=q.numel(),key.numel();ref=apply_rotary_pos_emb(q,key,cos,sin)
        data={'pytorch':bench(lambda:apply_rotary_pos_emb(q,key,cos,sin))}
        for block in (256,512,1024,2048):
            qb=triton.cdiv(nq,block)
            fn=lambda:k._rope[(qb+triton.cdiv(nk,block),)](q,key,cos,sin,oq,ok,nq,nk,12,2,t,128,*q.stride()[:3],*key.stride()[:3],cos.stride(0),cos.stride(1),qb,block,enable_fp_fusion=False)
            fn();diff=max(float((oq-ref[0]).abs().max()),float((ok-ref[1]).abs().max()))
            data[str(block)]={**bench(fn),'max_absolute_difference':diff}
        result['cases']['rope_'+str(batch)+'x'+str(t)]=data
        print('rope',batch,t,{x:round(v['median_us'],3)for x,v in data.items()},flush=True)
        (OUT/'kernel-launch-sweep.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()