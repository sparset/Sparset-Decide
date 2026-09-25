import sys,json,types,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import DecisionEngine,MODEL,DATA,question_from_dict,torch
from subset.engine import SYSTEM
from subset.schema import Choice,Noul
OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'
def message(e,context,q):
 if q.kind=='noul':end='\nReturn only true or false.'
 else:end='\nOptions:\n'+'\n'.join(f'{e.symbols[i]}. {json.dumps(l)}: {json.dumps(d)}' for i,(l,d) in enumerate(q.options))+'\nReturn only the letter of the best option.'
 return 'Context (JSON-encoded text):\n'+json.dumps(context)+'\n\nQuestion:\n'+q.instructions+end
examples=[('The box contains an apple.',Choice('Which kind of item is in the box?',{'fruit':'An edible fruit','rock':'A piece of stone','tool':'An instrument for work'}),'A'),('The box contains a hammer.',Choice('Which kind of item is in the box?',{'fruit':'An edible fruit','tool':'An instrument for work','rock':'A piece of stone'}),'B'),('The box contains a pebble.',Choice('Which kind of item is in the box?',{'fruit':'An edible fruit','tool':'An instrument for work','rock':'A piece of stone'}),'C'),('The lamp is switched on.',Noul('Is the lamp switched on?'),'true'),('The lamp is switched off.',Noul('Is the lamp switched on?'),'false')]
def demo_prepare(e,context,qs):
 common=[{'role':'system','content':SYSTEM}]
 for ctx,q,answer in examples:common.extend([{'role':'user','content':message(e,ctx,q)},{'role':'assistant','content':answer}])
 return [e.tokenizer.apply_chat_template(common+[{'role':'user','content':message(e,context,q)}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for q in qs.values()]
if __name__=='__main__':
 chosen=set(json.loads((OUT/'development-case-ids.json').read_text()));e=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,shared_attention='off');e.warmup();original=e.prepare
 with (OUT/'calibration-experiments.jsonl').open('w') as f:
  for variant in ['demos','prior_correction']:
   e.prepare=types.MethodType(demo_prepare,e) if variant=='demos' else original;n=0;priors={}
   for c in DATA['cases']:
    if c['id'] not in chosen:continue
    qs={k:question_from_dict(v) for k,v in c['questions'].items()};r=e.decide(c['context'],qs);ps=r['answers']['decision']['probabilities']
    if variant=='prior_correction':
     key=json.dumps(c['questions']);prior=priors.get(key)
     if prior is None:
      prior=e.decide('N/A',qs)['answers']['decision']['probabilities'];priors[key]=prior
     ps={k:p/max(prior[k],1e-20) for k,p in ps.items()};total=sum(ps.values());ps={k:p/total for k,p in ps.items()}
    ok=max(ps,key=ps.get)==c['expected']['decision'];n+=ok
    f.write(json.dumps({'case_id':c['id'],'variant':variant,'correct':ok,'probabilities':ps})+'\n');f.flush()
   print(variant,n,len(chosen),flush=True)
