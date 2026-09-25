import os,sys,json,time,shutil,hashlib,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import DecisionEngine,MODEL,OUT,DATA,question_from_dict,torch
DEST=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918';DEST.mkdir(exist_ok=True)
for p in (ROOT/'src/subset').glob('*.py'):
 target=DEST/'original-source'/p.name;target.parent.mkdir(exist_ok=True)
 if not target.exists():shutil.copy2(p,target)
rows=[json.loads(x) for x in (OUT/'quality.jsonl').read_text().splitlines()]
ids=['routing_001','routing_009','routing_007','routing_040','routing_030','boolean_013','boolean_031','boolean_042']
ids += [next(c['id'] for c in DATA['cases'] if c['group']==g) for g in ['banking','routing','boolean','scoring']]
cases=[c for c in DATA['cases'] if c['id'] in ids]
e=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,shared_attention='off')
e.warmup()
configs={'plain':(False,()),'graphs':(True,()),'fused':(False,('rmsnorm','swiglu','rope')),'both':(True,('rmsnorm','swiglu','rope'))}
with (DEST/'path-parity.jsonl').open('w') as f:
 for c in cases:
  q={k:question_from_dict(v) for k,v in c['questions'].items()}
  for mode,(graphs,fusions) in configs.items():
   e.clear_optimization_cache();gc.collect();torch.cuda.empty_cache();e.cuda_graphs=graphs;e.fused_kernels=fusions
   result=e.decide(c['context'],q)
   row={'case_id':c['id'],'expected':c['expected'],'mode':mode,'result':result}
   if mode=='plain':
    ids=e.prepare(c['context'],q)
    with torch.inference_mode():
     logits=e.model(input_ids=torch.tensor(ids,device='cuda'),use_cache=False).logits[0,-1].float()
     top=logits.softmax(-1).topk(6)
    row['raw_top']=[{'token':e.tokenizer.decode([i]),'p':p} for i,p in zip(top.indices.tolist(),top.values.tolist())]
    row['prompt']=e.tokenizer.decode(ids[0])
   f.write(json.dumps(row)+'\n');f.flush()
   print(c['id'],mode,result['answers'],flush=True)
e.clear_optimization_cache()
