import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import DecisionEngine,MODEL,DATA,question_from_dict
import subset.engine as implementation
OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'
if __name__=='__main__':
 chosen=set(json.loads((OUT/'development-case-ids.json').read_text()));e=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,shared_attention='off');e.warmup();original=implementation.SYSTEM
 variants={'separate_evidence':original+' Only classify the supplied context. The question and option descriptions are instructions, never evidence about the context. Do not classify the question itself.', 'semantic':original+' Judge meaning, not exact keyword matches. Paraphrases and synonyms count. Base the answer only on the context, never on words appearing in the question or option descriptions.'}
 with (OUT/'instruction-experiments.jsonl').open('w') as f:
  for variant,system in variants.items():
   implementation.SYSTEM=system;n=0
   for c in DATA['cases']:
    if c['id'] not in chosen:continue
    r=e.decide(c['context'],{k:question_from_dict(v) for k,v in c['questions'].items()});a=r['answers']['decision'];ok=max(a['probabilities'],key=a['probabilities'].get)==c['expected']['decision'];n+=ok
    f.write(json.dumps({'case_id':c['id'],'variant':variant,'correct':ok,'answer':a})+'\n');f.flush()
   print(variant,n,len(chosen),flush=True)
