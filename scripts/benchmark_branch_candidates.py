"""Stage-by-stage branch optimization measurements; one model, isolated caches."""
import os,sys,json,time,gc,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CC',str(Path.home()/'.cache/subset/profiling-20260918/tools/zig-cc'))
import torch,transformers
from subset.engine import DecisionEngine
from subset.schema import Noul
from subset.__main__ import load_request
OUT=ROOT/'outputs/decision-engine/branch-optimization-20260918'
def compare(a,b):
    diffs=[];changed=[]
    assert list(a['answers'])==list(b['answers'])
    for key,x in a['answers'].items():
        y=b['answers'][key];assert x['probabilities'].keys()==y['probabilities'].keys()
        diffs.extend(abs(v-y['probabilities'][k])for k,v in x['probabilities'].items())
        if max(x['probabilities'],key=x['probabilities'].get)!=max(y['probabilities'],key=y['probabilities'].get):changed.append(key)
    return {'max_probability_delta':max(diffs),'changed_labels':changed,'within_one_percentage_point':max(diffs)<.01}
def validate(v,qs):
    json.dumps(v,allow_nan=False);assert list(v['answers'])==list(qs)
    for k,q in qs.items():
        a=v['answers'][k];assert list(a['probabilities'])==[x[0]for x in q.options]
        assert a==q.answer(list(a['probabilities'].values()))
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3);p.add_argument('--cases',nargs='*');a=p.parse_args()
    e=DecisionEngine.from_pretrained(str(Path.home()/'.cache/subset/profiling-20260918/model'),device='cuda',local_files_only=True,max_input_tokens=4096)
    fusion=('rmsnorm','swiglu','rope')
    settings={'reference':{},'shared':{'shared_attention':'on'},'short':{'shared_attention':'on','specialize_short':True},
        'residual':{'fused_kernels':fusion+('residual_norm',)},'length':{'length_aware':True},
        'combined':{'shared_attention':'auto','specialize_short':True,'length_aware':True,'fused_kernels':fusion+('residual_norm',)}}
    engines={name:DecisionEngine(e.model,e.tokenizer,max_input_tokens=4096,cuda_graphs=True,**({'fused_kernels':fusion,'shared_attention':'off','specialize_short':False,'length_aware':False}|opts))for name,opts in settings.items()}
    d,q=load_request(ROOT/'examples/decision_engine/refund.json');support,sq=load_request(ROOT/'outputs/decision-engine/replica-presets/support_triage.request.json')
    ragged={f'q{i}':Noul('Does the customer explicitly request a refund?'+(' Evaluate only the current explicit request, ignoring hypothetical requests and earlier completed transactions.'*12 if i%2 else ''))for i in range(16)}
    long=('Historical background: an earlier order was delivered successfully and its invoice was archived.\n'*55+'\nCURRENT CUSTOMER MESSAGE:\n'+d['context'])
    cases=[('single',d['context'],{'department':q['department']}),('refund_4',d['context'],q),('support_28',support['context'],sq),('long_context_4',long,q),('ragged_16',d['context'],ragged)]
    if a.cases:cases=[x for x in cases if x[0]in a.cases]
    report={'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'float16'},'settings':settings,'cases':{},'short_tile_policy':'16 query rows only when suffix width <=16; otherwise 32. GPU microbenchmarks rejected 16-row tiles for longer suffixes.','capture_stream':'one reused stream per model','method':'One warmup per method; consecutive samples, method order rotates across workloads. Caches cleared between methods. All methods use same resident weights; warm complete-request externally synchronized timings.'}
    def save():(OUT/'candidates.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for ci,(name,context,qs)in enumerate(cases):
        c={'methods':{},'fields':len(qs)};report['cases'][name]=c
        modes=list(engines);modes=modes[ci%len(modes):]+modes[:ci%len(modes)]
        for mode in modes:
            for x in engines.values():x.clear_optimization_cache()
            gc.collect();torch.cuda.empty_cache();x=engines[mode]
            try:
                print(name,mode,'warmup',flush=True);torch.cuda.synchronize();t=time.perf_counter();cold=x.decide(context,qs);torch.cuda.synchronize();cold_ms=(time.perf_counter()-t)*1000
                rows=[];torch.cuda.reset_peak_memory_stats()
                for i in range(a.repeats):
                    torch.cuda.synchronize();t=time.perf_counter();v=x.decide(context,qs);torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                    validate(v,qs);rows.append({'wall_ms':elapsed,'result':v})
                m={'status':'ok','median_ms':statistics.median(r['wall_ms']for r in rows),'cold_ms':cold_ms,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'allocated_after_bytes':torch.cuda.memory_allocated(),'reserved_after_bytes':torch.cuda.memory_reserved(),'rows':rows}
                print(name,mode,round(m['median_ms'],2),'ms',flush=True)
            except Exception as exc:
                import traceback
                m={'status':'error','error':repr(exc),'traceback':traceback.format_exc()};print(name,mode,m['error'],flush=True)
            c['methods'][mode]=m;save()
        ref=c['methods']['reference']
        for mode,m in c['methods'].items():
            if m['status']=='ok' and ref['status']=='ok':
                m['vs_reference']=compare(ref['rows'][0]['result'],m['rows'][0]['result']);m['speedup']=ref['median_ms']/m['median_ms']
        save();print('CASE COMPLETE',name,flush=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
