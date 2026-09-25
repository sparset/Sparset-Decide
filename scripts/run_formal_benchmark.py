"""Frozen same-device quality, performance, and fresh-process benchmark."""
import time
PROCESS_ENTRY=time.perf_counter()
import os,sys,json,math,statistics,hashlib,platform,gc,traceback,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/formal-benchmark-20260918'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'outputs/decision-engine/comparison-20260918/replica-source')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CC',str(Path.home()/'.cache/subset/profiling-20260918/tools/zig-cc'))
import torch,transformers
from subset.engine import DecisionEngine
from subset.schema import question_from_dict
from core.schema import StructuredSchema
import core.engine_torch as hf
MODEL=str(Path.home()/'.cache/subset/profiling-20260918/model')
DATA=json.loads((OUT/'cases.json').read_text())

def unique_object(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('Duplicate JSON key: '+k)
        result[k]=v
    return result

def validate(answers,questions):
    if not isinstance(answers,dict) or set(answers)!=set(questions):raise ValueError('Field names do not match')
    for name,q in questions.items():
        a=answers[name]
        if not isinstance(a,dict) or set(a)!={'choice','probabilities'}:raise ValueError('Each answer needs choice and probabilities only')
        ps=a['probabilities'];labels=dict(q.options)
        if not isinstance(ps,dict) or set(ps)!=set(labels):raise ValueError('Probability labels do not match')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in ps.values()):raise ValueError('Invalid probability value')
        if abs(sum(ps.values())-1)>0.001:raise ValueError('Probability sum is not one within 0.001 rounding tolerance')
        if a['choice'] not in labels or ps[a['choice']]<max(ps.values())-1e-7:raise ValueError('Choice must maximize its probabilities')
    return True

def environment():
    return {'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'precision':'FP16 backbone; our selected head FP32','model':'Qwen/Qwen2.5-1.5B-Instruct','model_revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','replica_revision':'2af86848be75847ccb3553b0941cc51d6ef7e4e9','cases_sha256':hashlib.sha256((OUT/'cases.json').read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/subset').glob('*.py')}}

def make_schema(questions):
    return StructuredSchema({k:{'type':'boolean' if q.kind=='noul' else 'enum','description':q.instructions+(' Options: '+json.dumps(dict(q.options),ensure_ascii=False) if q.kind!='noul' else ''),**({} if q.kind=='noul' else {'choices':list(dict(q.options))})} for k,q in questions.items()})

class Runner:
    def __init__(self,method=None):
        self.engine=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,max_input_tokens=2048,prompt_format='readable',answer_encoding='letters',cuda_graphs=True,fused_kernels=('rmsnorm','swiglu','rope'))
        self.model=self.engine.model;self.tokenizer=self.engine.tokenizer
        hf._torch_model=self.model;hf._torch_tokenizer=self.tokenizer;hf._torch_device='cuda'
        self.schemas={}
    def call(self,method,case):
        q={k:question_from_dict(v) for k,v in case['questions'].items()}
        context=case['context'];details={};answers=None
        if method=='ours':
            native=self.engine.decide(context,q)
            answers={k:{'choice':max(a['probabilities'],key=a['probabilities'].get),'probabilities':a['probabilities']} for k,a in native['answers'].items()}
            details={'native':native}
        elif method=='hf':
            key=json.dumps(case['questions'],sort_keys=False)
            if key not in self.schemas:self.schemas[key]=make_schema(q)
            schema=self.schemas[key];meta=schema.compile_parallel_metadata(self.tokenizer)
            vectors=[];original=hf.F.softmax
            def capture(x,*args,**kwargs):
                y=original(x,*args,**kwargs)
                if x.ndim==1 and not x.is_cuda:vectors.append(y.detach().tolist())
                return y
            # Read-only observation of the actual score vector before upstream
            # truncates telemetry to its top five choices. No score modification.
            hf.F.softmax=capture
            try:native=hf.run_parallel_generation_torch(context,schema,temperature=1.0)
            finally:hf.F.softmax=original
            if len(vectors)!=len(q):raise RuntimeError('Could not capture all native HF probability vectors')
            answers={}
            for (k,question),vector in zip(q.items(),vectors):
                labels=['yes','no'] if question.kind=='noul' else list(dict(question.options))
                probs=dict(zip(labels,vector));value=native['parsed_json'][k]['value']
                choice=('yes' if value else 'no') if question.kind=='noul' else str(value)
                answers[k]={'choice':choice,'probabilities':probs}
            details={'native':native,'candidate_token_collisions':dict(zip(q,meta['has_collisions'])),'candidate_token_ids':dict(zip(q,meta['cands_per_field']))}
        else:
            label_only=method=='base_labels'
            fields={k:{'question':question.instructions,'options':dict(question.options)} for k,question in q.items()}
            if label_only:
                instruction='Evaluate each field against the context. Return only a JSON object mapping every field name to its chosen option label string. No markdown or explanation.'
            else:
                instruction=('Evaluate each field against the context. Return only one JSON object mapping every field name to an object with exactly choice and probabilities. '
                 'choice must be one allowed option label. probabilities must map EVERY allowed option label to a number between 0 and 1, summing to 1. '
                 'The chosen label must have the highest probability. Include all fields. No markdown, comments, or explanation. '
                 'Example format for a field with labels x and y: {"field":{"choice":"x","probabilities":{"x":0.8,"y":0.2}}}. Use the actual requested names and labels.')
            ids=self.tokenizer.apply_chat_template([{'role':'system','content':instruction},{'role':'user','content':json.dumps({'context':context,'fields':fields},ensure_ascii=False)}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            expected={k:({'choice':list(dict(question.options))[0],'probabilities':{label:round(1/len(question.options),4) for label,_ in question.options}} if not label_only else list(dict(question.options))[0]) for k,question in q.items()}
            cap=max(256,min(4096,2*len(self.tokenizer.encode(json.dumps(expected)))+64))
            if len(ids)+cap>self.model.config.max_position_embeddings:raise ValueError('Baseline context plus output budget exceeds model limit')
            inp=torch.tensor([ids],device='cuda')
            with torch.inference_mode():
                output=self.model.generate(inp,attention_mask=torch.ones_like(inp),do_sample=False,max_new_tokens=cap,use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
            tokens=output[0,len(ids):].cpu().tolist();text=self.tokenizer.decode(tokens,skip_special_tokens=True)
            eos=self.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            truncated=len(tokens)>=cap and (not tokens or tokens[-1] not in eos)
            details={'raw_text':text,'input_tokens':len(ids),'generated_tokens':len(tokens),'output_budget':cap,'truncated':truncated}
            try:
                answers=json.loads(text,object_pairs_hook=unique_object)
                if label_only:
                    if not isinstance(answers,dict) or set(answers)!=set(q):raise ValueError('Wrong label-only fields')
                    for k,v in answers.items():
                        if not isinstance(v,str) or v not in dict(q[k].options):raise ValueError('Invalid label-only answer')
                    return {'status':'valid' if not truncated else 'truncated','labels':answers,**details}
            except (ValueError,TypeError) as exc:return {'status':'truncated' if truncated else 'invalid','error':str(exc),**details}
        try:validate(answers,q)
        except (ValueError,TypeError,KeyError) as exc:return {'status':'invalid','error':str(exc),'answers':answers,**details}
        if details.get('truncated'):return {'status':'truncated','answers':answers,**details}
        return {'status':'valid','answers':answers,**details}
    def measured(self,method,case):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        try:r=self.call(method,case);torch.cuda.synchronize()
        except (torch.OutOfMemoryError,ValueError,RuntimeError) as exc:
            r={'status':'error','error':str(exc),'error_type':type(exc).__name__}
            self.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
        return {'case_id':case['id'],'method':method,'wall_ms':(time.perf_counter()-started)*1000,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'result':r}

def append(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def existing(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []

def timing_cases():
    q={'department':{'type':'choice','instructions':'Which department handles the customer issue?','criteria':{'billing':'Payment errors and duplicate charges','shipping':'Missing parcel delivery','returns':'Physical product returns or exchanges'}},'refund':{'type':'noul','instructions':'Does the current customer explicitly request a refund?'},'severity':{'type':'score','instructions':'Grade the reported impact.','criteria':['Information only with no loss','A billing error affecting one customer','A complete outage affecting all customers']}}
    base='Current customer message: My card was charged twice for one order. Please refund the duplicate charge. Only my account is affected. '
    filler='Background archive: The business sells household items online. Earlier informational notes describe shipping regions and opening hours. These notes report no current incidents. '
    cases=[]
    for fields in [1,4,16,28]:
        for size,repeats in [('short',0),('medium',10),('long',30)]:
            context=filler*repeats+base
            questions={f'field_{i:02d}':list(q.values())[i%3] for i in range(fields)}
            cases.append({'id':f'speed_{fields}_{size}','context':context,'questions':questions})
    return cases

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','full','speed','cold','labels'],required=True);p.add_argument('--method',choices=['ours','hf','base'],default='ours');p.add_argument('--trial',type=int,default=0);a=p.parse_args()
    if a.stage=='cold':
        r=Runner();loaded=time.perf_counter()-PROCESS_ENTRY
        if a.method=='ours':r.engine.warmup()
        ready=time.perf_counter()-PROCESS_ENTRY
        row=r.measured(a.method,DATA['cases'][0]);row.update({'stage':'cold','trial':a.trial,'model_loaded_seconds':loaded,'ready_seconds':ready,'process_entry_to_answer_seconds':time.perf_counter()-PROCESS_ENTRY,'environment':environment()})
        append(OUT/'cold.jsonl',row);print(json.dumps({'cold':a.method,'trial':a.trial,'seconds':row['process_entry_to_answer_seconds'],'status':row['result']['status']}),flush=True);return
    r=Runner();(OUT/'environment.json').write_text(json.dumps(environment(),indent=2)+'\n',encoding='utf-8')
    r.engine.warmup()
    if a.stage in ['pilot','full','labels']:
        path=OUT/('labels.jsonl' if a.stage=='labels' else 'quality.jsonl')
        done={(x['case_id'],x['method']) for x in existing(path)}
        cases=DATA['cases'][:20] if a.stage=='pilot' else DATA['cases'] if a.stage=='labels' else DATA['cases']+DATA['robustness_cases']
        for i,case in enumerate(cases):
            methods=['base_labels'] if a.stage=='labels' else ['ours','hf','base']
            methods=methods[i%len(methods):]+methods[:i%len(methods)]
            for method in methods:
                if (case['id'],method) in done:continue
                # Keep other methods' graph buffers from reducing baseline memory.
                r.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
                row=r.measured(method,case);row.update({'group':case['group'],'expected':case['expected']});append(path,row)
                print(f'{a.stage} {i+1}/{len(cases)} {case["id"]} {method} {row["wall_ms"]:.1f}ms {row["result"]["status"]}',flush=True)
    else:
        path=OUT/'speed.jsonl';done={(x['case_id'],x['method'],x['repeat']) for x in existing(path)}
        for ci,case in enumerate(timing_cases()):
            methods=['ours','hf','base'];methods=methods[ci%3:]+methods[:ci%3]
            for method in methods:
                if all((case['id'],method,j) in done for j in range(6)):continue
                r.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
                for j in range(6):
                    row=r.measured(method,case);row.update({'repeat':j,'phase':'first_shape' if j==0 else 'warm','fields':len(case['questions'])});append(path,row)
                    print(f'speed {case["id"]} {method} {j} {row["wall_ms"]:.1f}ms {row["result"]["status"]}',flush=True)
                    if row['result']['status']=='error':break
    print('STAGE COMPLETE',a.stage,flush=True)
if __name__=='__main__':main()
