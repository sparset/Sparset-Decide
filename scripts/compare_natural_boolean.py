"""Same-GPU task comparison: ordinary Qwen JSON, direct scoring, upstream HF, optimized engine."""
import os,sys,json,time,statistics,gc,hashlib,platform,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/natural-boolean-20260918'
SOURCE=ROOT/'outputs/decision-engine/comparison-20260918/replica-source'
sys.path[:0]=[str(ROOT/'src'),str(SOURCE)]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('CC',str(Path.home()/'.cache/sparset-decide/profiling-20260918/tools/zig-cc'))
import torch,transformers
from sparset_decide.engine import DecisionEngine
from sparset_decide.__main__ import load_request
from sparset_decide.benchmark import json_baseline
from core.schema import StructuredSchema
import core.engine_torch as hf
import importlib.util
spec=importlib.util.spec_from_file_location('sparset_decide.engine_before',OUT/'engine_before.py')
old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
OldDecisionEngine=old_module.DecisionEngine


def validate_output(result, questions):
    import math
    json.dumps(result,allow_nan=False)
    assert set(result['answers'])==set(questions)
    for key,q in questions.items():
        answer=result['answers'][key];p=answer['probabilities']
        assert list(p)==[label for label,_ in q.options]
        assert all(math.isfinite(v) and 0<=v<=1 for v in p.values())
        assert math.isclose(sum(p.values()),1,abs_tol=1e-5)
        assert answer==q.answer(list(p.values()))

REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

def schema_for(questions):
    return StructuredSchema({name:{'type':'boolean'if q.kind=='noul'else'enum',
        'description':q.instructions+(' Options: '+json.dumps(dict(q.options))if q.kind!='noul'else''),
        **({}if q.kind=='noul'else{'choices':[x[0]for x in q.options]})}for name,q in questions.items()})

def own_labels(result):
    return {k:max(v['probabilities'],key=v['probabilities'].get)for k,v in result['answers'].items()}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    model=str(Path.home()/'.cache/sparset-decide/profiling-20260918/model')
    e=DecisionEngine.from_pretrained(model,device='cuda',local_files_only=True,max_input_tokens=4096,
        cuda_graphs=True,fused_kernels=('rmsnorm','swiglu','rope'))
    plain=DecisionEngine(e.model,e.tokenizer,efficient_cache=False,max_input_tokens=4096)
    old=OldDecisionEngine(e.model,e.tokenizer,max_input_tokens=4096,cuda_graphs=True,fused_kernels=('rmsnorm','swiglu','rope'))
    # Normalize hardware, checkpoint and precision via upstream's existing model cache.
    # No changes to the downloaded inference or schema code.
    hf._torch_model=e.model;hf._torch_tokenizer=e.tokenizer;hf._torch_device='cuda'
    refund,rq=load_request(ROOT/'examples/decision_engine/refund.json')
    support,sq=load_request(ROOT/'outputs/decision-engine/replica-presets/support_triage.request.json')
    original_support=json.loads((ROOT/'outputs/decision-engine/replica-presets/support_triage.original.json').read_text())
    workloads=[('refund_1',refund,{'department':rq['department']},None),('refund_4',refund,rq,None),
               ('support_28',support,sq,StructuredSchema(original_support['schema']))]
    report={'environment':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'dtype':str(e.head.weight.dtype),'model':'Qwen/Qwen2.5-1.5B-Instruct','model_revision':REVISION},
        'replica_revision':'2af86848be75847ccb3553b0941cc51d6ef7e4e9','repeats':a.repeats,'cases':{},'change':'Boolean false/true scoring and shared format-neutral system instruction; other questions retain option IDs. Old engine is an unchanged pre-edit snapshot.', 'long_generation_sampling':'One measured 768-token run after a 16-token warmup, matching the previous completed long baseline protocol.',
        'method':'Warm full-request external synchronized wall time; one warmup and consecutive repetitions per method; graph buffers cleared between methods. Method order rotates across workloads. Same resident model weights/precision/GPU; different algorithms and native prompts. No inference implementation edits to HF replica.',
        'limitations':['Ordinary Qwen generates labels only; decision engines return probabilities as well. Prompts differ.',
            'HF upstream default CUDA loader may choose BF16; shared FP16 weights are injected for precision control.',
            'The refund schema is converted to HF types; option descriptions included in field descriptions. Support uses the pinned original upstream schema.',
            'Only the four-field refund case has tiny smoke labels; agreement is not accuracy.',
            'Graph/JIT cold costs and model loading are excluded; optimized graph speed requires reuse of matching shapes.']}
    def save(): (OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for wi,(name,data,questions,schema)in enumerate(workloads):
        schema=schema or schema_for(questions)
        meta=schema.compile_parallel_metadata(e.tokenizer)
        collisions={field[0]:{'choices':field[1].choices,'candidate_token_ids':ids}for field,ids,flag in zip(meta['field_items'],meta['cands_per_field'],meta['has_collisions'])if flag}
        case={'fields':len(questions),'context':data['context'],'hf_candidate_collisions':collisions,'methods':{}}
        report['cases'][name]=case;save()
        modes=['qwen_json','qwen_independent_scoring','hf_parallel','our_previous','our_optimized'];modes=modes[wi:]+modes[:wi]
        for mode in modes:
            e.clear_optimization_cache();plain.clear_optimization_cache();old.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
            def call():
                if mode=='qwen_json':return json_baseline(plain,data,questions,768)
                if mode=='qwen_independent_scoring':return plain.decide(data['context'],questions,mode='independent')
                if mode=='our_previous':return old.decide(data['context'],questions)
                if mode=='hf_parallel':return hf.run_parallel_generation_torch(data['context'],schema,temperature=1.0)
                return e.decide(data['context'],questions)
            rows=[];error=None
            print(name,mode,'warmup',flush=True)
            try:
                torch.cuda.synchronize();start=time.perf_counter();warm=json_baseline(plain,data,questions,16) if name=='support_28' and mode=='qwen_json' else call();torch.cuda.synchronize();warm_ms=(time.perf_counter()-start)*1000
                torch.cuda.reset_peak_memory_stats()
                for i in range(1 if name=='support_28' and mode=='qwen_json' else a.repeats):
                    torch.cuda.synchronize();start=time.perf_counter();v=call();torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    if mode=='qwen_json':labels=v.get('labels')if v['status']=='valid'else None
                    elif mode=='hf_parallel':labels={k:('yes'if x['value']else'no')if isinstance(x['value'],bool)else str(x['value'])for k,x in v['parsed_json'].items()}
                    else:
                        validate_output(v,questions)
                        labels=own_labels(v)
                    expected={k:x for k,x in data.get('expected_labels',{}).items()if k in questions}
                    quality={'correct':sum(labels.get(k)==x for k,x in expected.items()),'fields':len(expected)}if expected and labels is not None else None
                    rows.append({'wall_ms':elapsed,'labels':labels,'quality':quality,'result':v})
                    case['methods'][mode]={'status':'running','rows':rows};save()
                    print(name,mode,i+1,round(elapsed,2),'ms',flush=True)
                m={'status':'ok','warmup_ms':warm_ms,'schema_checked':mode in ('our_optimized','our_previous','qwen_independent_scoring'),'median_ms':statistics.median(x['wall_ms']for x in rows),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rows':rows}
            except Exception as exc:
                m={'status':'error','error':repr(exc),'traceback':traceback.format_exc(),'rows':rows}
                print(name,mode,'ERROR',repr(exc),flush=True)
            case['methods'][mode]=m;save()
        ours=case['methods'].get('our_optimized',{})
        if ours.get('status')=='ok':
            case['speedup_vs']={k:v['median_ms']/ours['median_ms']for k,v in case['methods'].items()if v['status']=='ok'and k!='our_optimized'}
            ol=ours['rows'][0]['labels']
            case['label_agreement_with_ours']={k:{'matching':sum(ol[f]==v['rows'][0]['labels'].get(f)for f in ol),'fields':len(ol)}for k,v in case['methods'].items()if v['status']=='ok'and v['rows'][0]['labels']is not None}
        save()
    e.clear_optimization_cache();old.clear_optimization_cache();print('COMPLETE',OUT/'comparison.json',flush=True)
if __name__=='__main__':main()