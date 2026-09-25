import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import DecisionEngine,MODEL,DATA,question_from_dict
from subset.schema import Noul
OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'
if __name__=='__main__':
 chosen=set(json.loads((OUT/'development-case-ids.json').read_text()));e=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,shared_attention='off');e.warmup()
 with (OUT/'binary-experiments.jsonl').open('w') as f:
  n=0
  for c in DATA['cases']:
   if c['id'] not in chosen:continue
   q=question_from_dict(c['questions']['decision'])
   if q.kind=='noul':qs={'yes':Noul('Read the context carefully. '+q.instructions)}
   else:
    qs={label:Noul('Task: '+q.instructions+'\nAll options: '+json.dumps(dict(q.options))+'\nProposed answer: '+json.dumps(label)+': '+desc+'\nIs this proposed answer the best match for the context?') for label,desc in q.options}
   r=e.decide(c['context'],qs)
   if q.kind=='noul':ps=r['answers']['yes']['probabilities']
   else:
    ps={k:a['noul'] for k,a in r['answers'].items()};total=sum(ps.values());ps={k:p/total for k,p in ps.items()}
   ok=max(ps,key=ps.get)==c['expected']['decision'];n+=ok
   f.write(json.dumps({'case_id':c['id'],'variant':'binary_options','correct':ok,'probabilities':ps,'metadata':r['metadata']})+'\n');f.flush()
  print('binary_options',n,len(chosen),flush=True)
