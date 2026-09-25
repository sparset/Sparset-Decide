"""Alternating confirmation of the automatic profile, including changed requests."""
from benchmark_branch_candidates import *
import statistics

def main():
    model=str(Path.home()/'.cache/subset/profiling-20260918/model')
    base=DecisionEngine.from_pretrained(model,device='cuda',local_files_only=True,max_input_tokens=4096)
    common={'max_input_tokens':4096,'cuda_graphs':True,'fused_kernels':('rmsnorm','swiglu','rope')}
    engines={
        'reference':DecisionEngine(base.model,base.tokenizer,**common,shared_attention='off',specialize_short=False,length_aware=False),
        'length_only':DecisionEngine(base.model,base.tokenizer,**common,shared_attention='off',specialize_short=False,length_aware=True),
        'automatic':DecisionEngine(base.model,base.tokenizer,**common),
    }
    d,q=load_request(ROOT/'examples/decision_engine/refund.json');support,sq=load_request(ROOT/'outputs/decision-engine/replica-presets/support_triage.request.json')
    ragged={f'q{i}':Noul('Does the customer explicitly request a refund?'+(' Evaluate only the current explicit request, ignoring hypothetical requests and earlier completed transactions.'*12 if i%2 else ''))for i in range(16)}
    long=('Historical background: an earlier order was delivered successfully and its invoice was archived.\n'*55+'\nCURRENT CUSTOMER MESSAGE:\n'+d['context'])
    terse={str(i):Noul(x)for i,x in enumerate(('Refund?','Human?','Outage?','Fraud?','Duplicate charge?','Shipping issue?','Account issue?','Cancellation?'))}
    cases=[('single',d['context'],{'department':q['department']}),('refund_4',d['context'],q),('support_28',support['context'],sq),('long_context_4',long,q),('ragged_16',d['context'],ragged),('terse_8',long,terse)]
    report={'method':'Five rounds per workload, rotating method order. Clear graph/head caches and warm each method before one timed sample, avoiding retained graph buffers from competitors. Model loading/JIT/capture excluded. Same weights and precision.','cases':{}}
    def save():(OUT/'confirmation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,context,qs in cases:
        c={'methods':{k:[]for k in engines},'fields':len(qs)};report['cases'][name]=c
        for i in range(5):
            modes=list(engines);modes=modes[i%3:]+modes[:i%3]
            for mode in modes:
                for e in engines.values():e.clear_optimization_cache()
                gc.collect();torch.cuda.empty_cache();e=engines[mode]
                e.decide(context,qs);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                v=e.decide(context,qs);torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                validate(v,qs);c['methods'][mode].append({'wall_ms':elapsed,'result':v,'peak_allocated_bytes':torch.cuda.max_memory_allocated()});save()
            print(name,'round',i+1,{m:round(c['methods'][m][-1]['wall_ms'],1)for m in engines},flush=True)
        c['medians']={m:statistics.median(x['wall_ms']for x in rows)for m,rows in c['methods'].items()}
        c['parity']={m:[compare(a['result'],b['result'])for a,b in zip(c['methods']['reference'],rows)]for m,rows in c['methods'].items()}
        for e in engines.values():e.clear_optimization_cache()
        changed=context.replace('refund','return').replace('Refund','Return')
        a=engines['reference'].decide(changed,qs);b=engines['automatic'].decide(changed,qs);validate(b,qs)
        c['changed_context_parity']=compare(a,b);save()
        print('CASE',name,c['medians'],c['changed_context_parity'],flush=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
