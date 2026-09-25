"""Paired end-to-end benchmarks against the saved pre-optimization engine."""
import argparse,sys,json,time,statistics,importlib.util,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from subset.engine import DecisionEngine
from subset.__main__ import load_request
import profile_engine as h
OUT=ROOT/'outputs/decision-engine/optimization-20260918'
spec=importlib.util.spec_from_file_location('subset._baseline',OUT/'baseline_engine.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--repeats',type=int,default=7);a=p.parse_args()
 model=str(Path.home()/'.cache/subset/profiling-20260918/model')
 e=DecisionEngine.from_pretrained(model,device='cuda',local_files_only=True)
 b=mod.DecisionEngine(e.model,e.tokenizer)
 stages={'cache':{},'graphs':{'cuda_graphs':True},'rmsnorm':{'fused_kernels':('rmsnorm',)},'swiglu':{'fused_kernels':('swiglu',)},'rope':{'fused_kernels':('rope',)},'fused':{'fused_kernels':('rmsnorm','swiglu','rope')},'combined':{'cuda_graphs':True,'fused_kernels':('rmsnorm','swiglu','rope')}}
 opts=stages[a.stage]
 if opts:e=DecisionEngine(e.model,e.tokenizer,**opts)
 refund,qs=load_request(ROOT/'examples/decision_engine/refund.json')
 support_path=ROOT/'outputs/decision-engine/replica-presets/support_triage.request.json'
 support,sq=load_request(support_path)
 long={'context':('Historical background: an earlier order was delivered successfully and its invoice was archived.\n'*55+'\nCURRENT CUSTOMER MESSAGE:\n'+refund['context'])};lq=qs
 workloads=[('refund_1',refund['context'],{'department':qs['department']} if 'department'in qs else {next(iter(qs)):next(iter(qs.values()))}),('refund_4',refund['context'],qs),('support_28',support['context'],sq),('long_context_4',long['context'],lq)]
 result={'stage':a.stage,'options':opts,'repeats':a.repeats,'cases':{},'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'scope':'Full requests, rotated baseline/variant order; warm capture/JIT excluded and cold timing reported separately. No labeled accuracy estimate.'}
 for name,context,questions in workloads:
  if hasattr(e,'clear_optimization_cache'):e.clear_optimization_cache()
  gc.collect();torch.cuda.empty_cache()
  rows={'baseline':[],'variant':[]};cold={}
  for label,engine in [('baseline',b),('variant',e)]:
   start=time.perf_counter();cold[label]=engine.decide(context,questions);cold[label]['external_ms']=(time.perf_counter()-start)*1000
   for _ in range(2):engine.decide(context,questions)
  torch.cuda.reset_peak_memory_stats()
  for i in range(a.repeats):
   for label,engine in ([('baseline',b),('variant',e)]if i%2==0 else [('variant',e),('baseline',b)]):
    start=time.perf_counter();v=engine.decide(context,questions);v['external_ms']=(time.perf_counter()-start)*1000;rows[label].append(v)
  med={label:statistics.median(v['external_ms'] for v in values)for label,values in rows.items()}
  comp=h.compare(rows['baseline'][0],rows['variant'][0]); changed=context.replace('refund','return')
  check=h.compare(b.decide(changed,questions),e.decide(changed,questions))
  case={'median_ms':med,'speedup':med['baseline']/med['variant'],'parity':comp,'changed_input_parity':check,'cold':cold,'rows':rows,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
  result['cases'][name]=case
  (OUT/(a.stage+'.json')).write_text(json.dumps(result,indent=2)+'\n')
  print(name,json.dumps({k:case[k]for k in ('median_ms','speedup','parity','changed_input_parity')}),flush=True)
 if hasattr(e,'clear_optimization_cache'):e.clear_optimization_cache()
if __name__=='__main__':main()