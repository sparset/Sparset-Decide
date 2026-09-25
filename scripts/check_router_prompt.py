import os,sys,json,itertools,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/decision-engine/huggingface'));os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from subset import Choice,Noul,Score
from subset.engine import DecisionEngine,SYSTEM
from subset.__main__ import load_request
OUT=ROOT/'outputs/decision-engine/router-diagnosis-20260918'
previous=json.loads((OUT/'windows.json').read_text())
class Readable(DecisionEngine):
    style='escaped'
    def prepare(self,context,questions):
        original=super().prepare(context,questions)
        rows=[]
        for i,q in enumerate(questions.values()):
            if q.kind=='noul':rows.append(original[i]);continue
            esc=(lambda x:json.dumps(x,ensure_ascii=False)) if self.style=='escaped' else (lambda x:x)
            options='\n'.join(f'{self.symbols[j]}. {esc(label)}: {esc(desc)}' for j,(label,desc) in enumerate(q.options))
            user='Context (JSON-encoded text):\n'+json.dumps(context,ensure_ascii=False)+'\n\nQuestion:\n'+q.instructions+'\nOptions:\n'+options+'\nReturn only the letter of the best option.'
            ids=self.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':user}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            assert len(ids)<=self.max_input_tokens
            rows.append(ids)
        return rows

def main():
    e=DecisionEngine.from_pretrained(device='cuda',local_files_only=True,prompt_format='json')
    report={'warmup':{},'cases':[],'permutations':[],'refund':{},'timings':{}}
    def save():(OUT/'prompt-experiments.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def measure(fn):
        torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return {'ms':(time.perf_counter()-t)*1000,'result':v}
    report['warmup']['initialization']=measure(lambda:e.decide('Initialization input.',{'ready':Noul('Is this text present?')}))
    context=previous['context'];prompt=previous['prompt'];criteria=previous['criteria'];questions={'model':Choice(prompt,criteria)}
    report['warmup']['first_real_request']=measure(lambda:e.decide(context,questions));save()
    print('warmup',report['warmup']['initialization']['ms'],'first real',report['warmup']['first_real_request']['ms'],flush=True)
    escaped=Readable(e.model,e.tokenizer);plain=Readable(e.model,e.tokenizer);plain.style='plain'
    methods={'original':e,'escaped':escaped,'plain':plain}
    for case in previous['diagnostic_cases']:
        row={'name':case['name'],'expected':case['expected'],'context':case['context'],'results':{}}
        for key,engine in methods.items():row['results'][key]=engine.decide(case['context'],questions)['answers']
        report['cases'].append(row);save()
    for order in itertools.permutations(criteria):
        q={'model':Choice(prompt,{k:criteria[k] for k in order})}
        row={'order':order,'results':{}}
        for key,engine in methods.items():row['results'][key]=engine.decide(context,q)['answers']
        report['permutations'].append(row);save()
        print('permutation',order,{k:v['model']['choice'] for k,v in row['results'].items()},flush=True)
    data,refund=load_request(ROOT/'examples/decision_engine/refund.json')
    for key,engine in methods.items():report['refund'][key]=engine.decide(data['context'],refund);save()
    # Additional tasks unrelated to routing, evaluated after choosing candidates.
    cases=[('sentiment_pos','I loved the service and will return!',Choice('What is the sentiment?',{'negative':'Unhappy or critical','neutral':'Neither positive nor negative','positive':'Happy or approving'}),'positive'),
           ('sentiment_neg','Terrible service. I will never come back.',Choice('What is the sentiment?',{'negative':'Unhappy or critical','neutral':'Neither positive nor negative','positive':'Happy or approving'}),'negative'),
           ('language','Bonjour, comment allez-vous?',Choice('What language is used?',{'English':'English language','French':'French language','Spanish':'Spanish language'}),'French'),
           ('shipping','My parcel never arrived; the tracking has not updated in a week.',refund['department'],'shipping'),
           ('billing','My credit card was charged twice for one order.',refund['department'],'billing'),
           ('returns','The shirt does not fit. I want to exchange it for a larger size.',refund['department'],'returns'),
           ('severity_low','Can you tell me the opening hours?',refund['severity'],'0'),
           ('severity_high','The service is down for every customer; nobody can use the product.',refund['severity'],'2')]
    report['other_tasks']=[]
    for name,ctx,q,expected in cases:
        row={'name':name,'context':ctx,'expected':expected,'results':{}}
        for key,engine in methods.items():row['results'][key]=engine.decide(ctx,{'answer':q})['answers']
        report['other_tasks'].append(row);save()
    for key,engine in methods.items():
        engine.decide(context,questions)
        report['timings'][key]=[measure(lambda:engine.decide(context,questions)) for _ in range(5)];save()
    print('DONE',flush=True)
if __name__=='__main__':main()
