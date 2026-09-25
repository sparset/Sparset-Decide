import os,sys,json,collections,faulthandler
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CC',str(Path.home()/'.cache/subset/profiling-20260918/tools/zig-cc'))
from profile_cupti import Cupti
faulthandler.cancel_dump_traceback_later()
from subset.engine import DecisionEngine
from subset.__main__ import load_request
import torch
out=ROOT/'outputs/decision-engine/branch-optimization-20260918'
e=DecisionEngine.from_pretrained(str(Path.home()/'.cache/subset/profiling-20260918/model'),device='cuda',local_files_only=True,cuda_graphs=True,fused_kernels=('rmsnorm','swiglu','rope'),shared_attention='off',specialize_short=False,length_aware=False)
c=Cupti();report={}
for name,path in [('refund_4','examples/decision_engine/refund.json'),('support_28','outputs/decision-engine/replica-presets/support_triage.request.json')]:
 d,q=load_request(ROOT/path);e.clear_optimization_cache()
 e.decide(d['context'],q);e.decide(d['context'],q)
 c.records=[];c.start();v=e.decide(d['context'],q);c.stop()
 counts=collections.defaultdict(lambda:[0,0.])
 for x in c.records:
  counts[x['name']][0]+=1;counts[x['name']][1]+=x['duration_us']or 0
 report[name]={'metadata':v['metadata'],'kernels':[{'name':k,'calls':x[0],'us':x[1]}for k,x in sorted(counts.items(),key=lambda x:-x[1][1])],'cupti_errors':c.errors,'dropped':c.dropped}
 (out/'profile-before.json').write_text(json.dumps(report,indent=2))
 print(name,[(x['name'][:65],round(x['us']/1000,2))for x in report[name]['kernels'][:10]],flush=True)
print('COMPLETE',flush=True)
