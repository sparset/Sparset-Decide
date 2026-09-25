import os,sys,json,time,statistics,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'outputs/decision-engine/comparison-20260918/replica-source')]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/decision-engine/huggingface'));os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from subset import Workflow,Choice
from subset.engine import DecisionEngine
from core.schema import StructuredSchema
import core.engine_torch as hf
OUT=ROOT/'outputs/decision-engine/router-diagnosis-20260918'
d=json.loads((OUT/'windows.json').read_text())
e=DecisionEngine.from_pretrained(device='cuda',local_files_only=True)
report={'platform':platform.platform(),'cpu_threads':torch.get_num_threads(),'warmup':e.warmup(),'methods':{}}
workflow=Workflow.load(ROOT/'examples/decision_engine/router.workflow.json')
context=d['context'];prompt=d['prompt'];criteria=d['criteria']
old=DecisionEngine(e.model,e.tokenizer,prompt_format='json')
graph=DecisionEngine(e.model,e.tokenizer,cuda_graphs=True)
hf._torch_model=e.model;hf._torch_tokenizer=e.tokenizer;hf._torch_device='cuda'
schema=StructuredSchema({'model':{'type':'enum','description':prompt+' Options: '+json.dumps(criteria),'choices':list(criteria)}})
methods={'old':lambda:workflow.run(old,context=context,prompt=prompt),
         'readable':lambda:workflow.run(e,context=context,prompt=prompt),
         'readable_graphs':lambda:workflow.run(graph,context=context,prompt=prompt),
         'hf_replica':lambda:hf.run_parallel_generation_torch(context,schema,temperature=1.0)}
def save():(OUT/'confirmation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def run(fn):
 torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return {'wall_ms':(time.perf_counter()-t)*1000,'result':v}
for name,fn in methods.items():
 report['methods'][name]={'first':run(fn),'rows':[]};save()
for i in range(5):
 names=list(methods);names=names[i%4:]+names[:i%4]
 for name in names:
  row=run(methods[name]);report['methods'][name]['rows'].append(row);save();print(i,name,round(row['wall_ms'],2),flush=True)
for name,value in report['methods'].items():
 value['median_ms']=statistics.median(r['wall_ms'] for r in value['rows'])
 print(name,value['median_ms'],value['rows'][0]['result'].get('answers',value['rows'][0]['result'].get('parsed_json')),flush=True)
report['graph_max_probability_delta']=max(abs(a['result']['answers']['model']['probabilities'][k]-b['result']['answers']['model']['probabilities'][k]) for a,b in zip(report['methods']['readable']['rows'],report['methods']['readable_graphs']['rows']) for k in criteria)
assert report['graph_max_probability_delta']<0.01
assert all(r['result']['answers']['model']['choice']=='coding' for name in ['readable','readable_graphs'] for r in report['methods'][name]['rows'])
save();graph.clear_optimization_cache()
