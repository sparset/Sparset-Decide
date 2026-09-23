"""Complete the long generation baseline with immediate per-sample checkpoints."""
import sys,json,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from sparset_decide.engine import DecisionEngine
from sparset_decide.__main__ import load_request
from sparset_decide.benchmark import json_baseline
OUT=ROOT/'outputs/decision-engine/comparison-20260918'
e=DecisionEngine.from_pretrained(str(Path.home()/'.cache/sparset-decide/profiling-20260918/model'),device='cuda',local_files_only=True,efficient_cache=False,max_input_tokens=4096)
data,qs=load_request(ROOT/'outputs/decision-engine/replica-presets/support_triage.request.json')
report={'case':'support_28','reason':'Initial suite reached its overall wall-time cap before saving the long generation method. This isolated follow-up checkpoints every result.','methods':{}}
def save(): (OUT/'long-baselines.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print('Short generation warmup: 16 output tokens',flush=True)
json_baseline(e,data,qs,16)
torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
v=json_baseline(e,data,qs,768);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
report['methods']['qwen_json']={'status':'ok','median_ms':elapsed,'sample_count':1,'warmup':'same prompt, 16 output tokens','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rows':[{'wall_ms':elapsed,'labels':v.get('labels')if v['status']=='valid'else None,'quality':None,'result':v}]};save()
print('Long JSON result',v['status'],v['generated_tokens'],round(elapsed,2),'ms',flush=True)
gc.collect();torch.cuda.empty_cache();e.decide(data['context'],qs,mode='independent');rows=[]
for i in range(3):
 torch.cuda.synchronize();start=time.perf_counter();v=e.decide(data['context'],qs,mode='independent');torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
 rows.append({'wall_ms':elapsed,'labels':{k:max(x['probabilities'],key=x['probabilities'].get)for k,x in v['answers'].items()},'quality':None,'result':v})
 import statistics
 report['methods']['qwen_independent_scoring']={'status':'ok','median_ms':statistics.median(x['wall_ms']for x in rows),'rows':rows};save();print('Independent',i+1,round(elapsed,2),'ms',flush=True)
print('COMPLETE',flush=True)