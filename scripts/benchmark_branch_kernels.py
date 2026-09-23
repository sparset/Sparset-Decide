"""Isolate GPU kernel benefits, separately from end-to-end measurements."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CC',str(Path.home()/'.cache/sparset-decide/profiling-20260918/tools/zig-cc'))
import torch
from sparset_decide.shared_attention import attention
from sparset_decide.kernels import residual_rmsnorm,rmsnorm
OUT=ROOT/'outputs/decision-engine/branch-optimization-20260918'
@torch.inference_mode()
def bench(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    # Amortize host replay overhead across 50 kernel invocations per graph.
    with torch.cuda.graph(graph):
        for _ in range(50):value=fn()
    graph.replay();torch.cuda.synchronize()
    samples=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(10):graph.replay()
        end.record();end.synchronize();samples.append(start.elapsed_time(end)/500)
    import statistics
    return statistics.median(samples)
@torch.inference_mode()
def main():
    torch.manual_seed(81);report={'short':[],'residual':[],'gpu':torch.cuda.get_device_name()}
    for t in (8,16,32,64,128):
        for p in (128,512,1024):
            q=torch.randn(8,12,t,128,device='cuda',dtype=torch.float16)
            k=torch.randn(8,2,t,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
            pk=torch.randn(1,2,p,128,device='cuda',dtype=torch.float16);pv=torch.randn_like(pk)
            lengths=torch.full((8,),t,device='cuda');fn=lambda short:attention(q,k,v,pk,pv,lengths,128**-.5,short)
            generic=bench(lambda:fn(False));short=bench(lambda:fn(True));delta=float((fn(False)-fn(True)).abs().max())
            row={'suffix':t,'prefix':p,'generic_ms':generic,'short_ms':short,'speedup':generic/short,'max_hidden_delta':delta}
            report['short'].append(row);print('short',row,flush=True)
    for rows in (8,32,128,512,1024):
        x=torch.randn(rows,1536,device='cuda',dtype=torch.float16);r=torch.randn_like(x);w=torch.randn(1536,device='cuda',dtype=torch.float16)
        def separate():
            added=x+r
            return added,rmsnorm(added,w,1e-6)
        baseline=bench(separate);fused=bench(lambda:residual_rmsnorm(x,r,w,1e-6))
        row={'rows':rows,'separate_ms':baseline,'fused_ms':fused,'speedup':baseline/fused};report['residual'].append(row);print('residual',row,flush=True)
    (OUT/'kernel-benchmarks.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
