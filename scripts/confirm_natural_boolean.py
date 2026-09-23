"""Alternating large-workload timing check and explicit positive/negative boolean controls."""
from compare_natural_boolean import *
from sparset_decide.schema import Noul

e=DecisionEngine.from_pretrained(str(Path.home()/'.cache/sparset-decide/profiling-20260918/model'),device='cuda',local_files_only=True,max_input_tokens=4096,cuda_graphs=True,fused_kernels=('rmsnorm','swiglu','rope'))
old=OldDecisionEngine(e.model,e.tokenizer,max_input_tokens=4096,cuda_graphs=True,fused_kernels=('rmsnorm','swiglu','rope'))
hf._torch_model=e.model;hf._torch_tokenizer=e.tokenizer;hf._torch_device='cuda'
data,questions=load_request(ROOT/'outputs/decision-engine/replica-presets/support_triage.request.json')
original=json.loads((ROOT/'outputs/decision-engine/replica-presets/support_triage.original.json').read_text(encoding='utf-8'))
schema=StructuredSchema(original['schema'])
report={'reason':'Initial large-workload new-engine samples ranged from 1.37 to 4.48 seconds; confirm with five alternating rounds and fresh warmup after clearing graph memory for each method.','timings':{k:[]for k in ('our_previous','our_optimized','hf_parallel')},'boolean_controls':[]}
def save(): (OUT/'confirmation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
for i in range(5):
    modes=['our_previous','our_optimized','hf_parallel'];modes=modes[i%3:]+modes[:i%3]
    for mode in modes:
        e.clear_optimization_cache();old.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
        def call():
            if mode=='hf_parallel':return hf.run_parallel_generation_torch(data['context'],schema,temperature=1.)
            return (e if mode=='our_optimized' else old).decide(data['context'],questions)
        call();torch.cuda.synchronize();start=time.perf_counter();v=call();torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        if mode!='hf_parallel':validate_output(v,questions)
        report['timings'][mode].append({'wall_ms':elapsed,'result':v});save();print('Confirm',i+1,mode,round(elapsed,2),flush=True)
e.clear_optimization_cache();old.clear_optimization_cache();e.cuda_graphs=False;old.cuda_graphs=False
qs={'refund_requested':Noul('Does the customer explicitly request a refund?'),'human_requested':Noul('Does the customer explicitly ask to speak to a human?')}
controls=[
 ('I would like a refund for the duplicate charge.', 'yes','no'),
 ('I do not want a refund. Please just send my receipt.', 'no','no'),
 ('I already received my refund. No additional refund is needed.', 'no','no'),
 ('Please connect me to a human agent. No refund is needed.', 'no','yes'),
 ('Please refund my purchase and put me through to a human agent.', 'yes','yes'),
 ('Where is my parcel? Do not transfer me to anyone.', 'no','no'),
 ('No refund please. I need a human agent to fix my login.', 'no','yes'),
 ('I want a refund, but automated assistance is fine. I do not need a human.', 'yes','no'),
]
for context,refund,human in controls:
    row={'context':context,'expected':{'refund_requested':refund,'human_requested':human},'methods':{}}
    for mode in ('our_previous','our_optimized','hf_parallel'):
        if mode=='hf_parallel':
            v=hf.run_parallel_generation_torch(context,schema_for(qs),temperature=1.)
            labels={k:'yes'if x['value']else'no'for k,x in v['parsed_json'].items()}
        else:
            v=(e if mode=='our_optimized'else old).decide(context,qs);validate_output(v,qs);labels=own_labels(v)
        row['methods'][mode]={'labels':labels,'correct':sum(labels[k]==x for k,x in row['expected'].items()),'result':v}
    report['boolean_controls'].append(row);save()
print('CONTROL SCORES', {m:sum(x['methods'][m]['correct']for x in report['boolean_controls'])for m in report['timings']},'/ 16',flush=True)
print('COMPLETE',flush=True)
