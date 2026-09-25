"""Reproduce a reported routing failure with pinned resident weights and upstream code."""
import os, sys, json, time, statistics, itertools, platform, argparse, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'outputs/decision-engine/comparison-20260918/replica-source')]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/decision-engine/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('CC',str(Path.home()/'.cache/subset/profiling-20260918/tools/zig-cc'))
import torch, transformers
from subset import Choice, Workflow
from subset.engine import DecisionEngine, SYSTEM
from core.schema import StructuredSchema
import core.engine_torch as hf

CONTEXT='Please code me an entire inference optimization engine that would work for any AI model of any size (LLMs only)'
PROMPT='What model should be routed for this request?'
CRITERIA=json.loads((ROOT/'examples/decision_engine/router.workflow.json').read_text())['questions']['model']['criteria']

def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');a=p.parse_args()
    tag='windows' if sys.platform=='win32' else 'linux'
    out=ROOT/'outputs/decision-engine/router-diagnosis-20260918';out.mkdir(parents=True,exist_ok=True)
    result={'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,
            'gpu':torch.cuda.get_device_name(),'default_cpu_threads':torch.get_num_threads(),
            'context':CONTEXT,'prompt':PROMPT,'criteria':CRITERIA,
            'replica_revision':'2af86848be75847ccb3553b0941cc51d6ef7e4e9',
            'method':'Same resident FP16 checkpoint, synchronized wall time; native HF prompt/algorithm unchanged. Schema field description includes exact question and all option descriptions. No calibration claims.'}
    def save(): (out/(tag+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def measured(fn):
        torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize()
        return {'wall_ms':(time.perf_counter()-t)*1000,'result':v}
    model='Qwen/Qwen2.5-1.5B-Instruct' if tag=='windows' else str(Path.home()/'.cache/subset/profiling-20260918/model')
    print('Loading',tag,flush=True)
    t=time.perf_counter();e=DecisionEngine.from_pretrained(model,device='cuda',local_files_only=True,prompt_format='json')
    result['load_seconds']=time.perf_counter()-t
    questions={'model':Choice(PROMPT,CRITERIA)}
    run=lambda:e.decide(CONTEXT,questions)
    result['first_request']=measured(run);save()
    print('first',round(result['first_request']['wall_ms'],1),result['first_request']['result']['answers'],flush=True)
    result['warm_default']=[measured(run) for _ in range(5)];save()
    print('warm default',round(statistics.median(x['wall_ms'] for x in result['warm_default']),1),flush=True)
    initial=torch.get_num_threads()
    result['threads']={}
    for n in dict.fromkeys([1,2,4,initial]):
        torch.set_num_threads(n);run()
        rows=[measured(run) for _ in range(3)]
        result['threads'][str(n)]={'median_ms':statistics.median(x['wall_ms'] for x in rows),'rows':rows};save()
        print('threads',n,result['threads'][str(n)]['median_ms'],flush=True)
    # Use one CPU thread in controlled comparisons; full model remains on GPU.
    torch.set_num_threads(1)
    optimized=DecisionEngine(e.model,e.tokenizer,cuda_graphs=True,prompt_format='json',
        fused_kernels=('rmsnorm','swiglu','rope') if tag=='linux' else ())
    hf._torch_model=e.model;hf._torch_tokenizer=e.tokenizer;hf._torch_device='cuda'
    def schema(prompt,criteria):
        return StructuredSchema({'model':{'type':'enum','description':prompt+' Options: '+json.dumps(criteria),'choices':list(criteria)}})
    hs=schema(PROMPT,CRITERIA)
    meta=hs.compile_parallel_metadata(e.tokenizer)
    result['hf_schema']={'description':hs.to_parallel_schema_str(),'candidate_ids':meta['cands_per_field'],'collisions':meta['has_collisions']}
    methods={'engine':run,'graphs':lambda:optimized.decide(CONTEXT,questions),
             'hf_replica':lambda:hf.run_parallel_generation_torch(CONTEXT,hs,temperature=1.0)}
    result['comparison']={}
    for name,fn in methods.items():
        result['comparison'][name]={'first_call':measured(fn),'rows':[]};save()
    for i in range(5):
        names=list(methods);names=names[i%3:]+names[:i%3]
        for name in names:
            row=measured(methods[name]);result['comparison'][name]['rows'].append(row)
            print('compare',i,name,round(row['wall_ms'],1),flush=True);save()
    optimized.clear_optimization_cache()
    if a.quick:
        save();return
    # Full-vocabulary logits verify the selected-row projection and inspect what
    # unconstrained generation would actually begin with on our exact template.
    ids=e.prepare(CONTEXT,questions)[0]
    with torch.inference_mode():
        inp=torch.tensor([ids],device='cuda')
        logits=e.model(inp,use_cache=False).logits[0,-1].float()
        values,indices=logits.softmax(-1).topk(15)
        result['raw_top_tokens']=[{'token':e.tokenizer.decode([i]),'id':i,'p':v} for i,v in zip(indices.tolist(),values.tolist())]
        full=logits[list(e._option_token_ids[:3])].softmax(-1).tolist()
        result['full_head_probabilities']=dict(zip(CRITERIA,full))
        generated=e.model.generate(inp,attention_mask=torch.ones_like(inp),max_new_tokens=24,do_sample=False,pad_token_id=e.tokenizer.eos_token_id)
        result['original_prompt_greedy_text']=e.tokenizer.decode(generated[0,len(ids):],skip_special_tokens=False)
    result['rendered_original_prompt']=e.tokenizer.decode(ids);save()
    result['permutations']=[]
    for order in itertools.permutations(CRITERIA):
        criteria={k:CRITERIA[k] for k in order}
        own=e.decide(CONTEXT,{'model':Choice(PROMPT,criteria)})
        other=hf.run_parallel_generation_torch(CONTEXT,schema(PROMPT,criteria),temperature=1.0)
        result['permutations'].append({'order':order,'ours':own['answers'],'hf':other});save()
    # Generic prompt experiments only; no special handling for the reported input.
    cases=[('reported',CONTEXT,'coding'),
        ('python','Write a Python function that parses CSV and validates each row.','coding'),
        ('sql','Debug this SQL query: SELECT name FROM users WHERE age = NULL;','coding'),
        ('greeting','Say hello in one short sentence.','fast'),
        ('capital','What is the capital of France?','fast'),
        ('translate','Translate hello into Spanish.','fast'),
        ('proof','Prove that there are infinitely many primes, explaining each logical step.','reasoning'),
        ('planning','Work out a detailed multi-stage research plan with dependencies and resource tradeoffs for investigating a new theory.','reasoning')]
    def variant(context,prompt,criteria,style):
        if style=='original':return e.decide(context,{'model':Choice(prompt,criteria)})['answers']['model']['probabilities']
        options=list(criteria)
        if style=='plain_ids':
            options_text='\n'.join(f'{chr(65+i)}. {k}: {criteria[k]}' for i,k in enumerate(options))
            ending='Return only the letter of the best option.';candidate_ids=list(e._option_token_ids[:len(options)])
        else:
            options_text='\n'.join(f'{k}: {criteria[k]}' for k in options)
            ending='Return only the exact option label.'
            encoded=[e.tokenizer.encode(k,add_special_tokens=False) for k in options]
            if any(len(x)!=1 for x in encoded):return {'unsupported_multitoken_labels':encoded}
            candidate_ids=[x[0] for x in encoded]
        text='Context:\n'+context+'\n\nQuestion: '+prompt+'\nOptions:\n'+options_text+'\n'+ending
        ids=e.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_dict=False)
        with torch.inference_mode():
            hidden,_=e._hidden([ids]);w=e.head.weight[candidate_ids].float()
            logits=torch.nn.functional.linear(hidden.float(),w)
            probs=logits.softmax(-1)[0].tolist()
        return dict(zip(options,probs))
    result['diagnostic_cases']=[]
    for name,context,expected in cases:
        record={'name':name,'context':context,'expected':expected,'variants':{}}
        for style in ['original','plain_ids','natural_labels']:
            record['variants'][style]=variant(context,PROMPT,CRITERIA,style)
        record['hf']=hf.run_parallel_generation_torch(context,schema(PROMPT,CRITERIA),temperature=1.0)
        result['diagnostic_cases'].append(record);save()
        print(name,record['variants'],flush=True)
    # CPU trace identifies launch/dispatch work; do not treat profiled latency as a benchmark.
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU],record_shapes=True) as prof:run()
    result['cpu_profile']=[{'name':x.key,'calls':x.count,'self_cpu_us':x.self_cpu_time_total,'total_cpu_us':x.cpu_time_total} for x in sorted(prof.key_averages(),key=lambda x:-x.self_cpu_time_total)[:25]]
    save();print('DONE',out/(tag+'.json'),flush=True)
if __name__=='__main__':main()
