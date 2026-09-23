"""Extended profiling experiments. No production engine code or saved weights change."""
import argparse
import copy
import types
import contextlib
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/decision-engine/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import transformers
import profile_engine as h
from profile_backend_diagnostic import routing
import sparset_decide.engine as engine_module
from sparset_decide.engine import DecisionEngine
from sparset_decide.__main__ import load_request
from torch.nn.attention import sdpa_kernel, SDPBackend

def save(p,v): h.save(p,v)

def spread(rows):
    first=rows[0]["answers"]
    return max(abs(a["probabilities"][o]-row["answers"][k]["probabilities"][o])
               for row in rows for k,a in first.items() for o in a["probabilities"])

@contextlib.contextmanager
def fp32_reference(engine):
    """FP32 arithmetic for the same FP16-rounded weights; only one decoder layer upcast at a time."""
    hooks=[]
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    old_cudnn=torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    engine.backbone.norm.float()
    # An embedding lookup has no arithmetic; upcasting its outputs preserves the exact stored values.
    hooks.append(engine.backbone.embed_tokens.register_forward_hook(lambda mod,args,out: out.float()))
    def before(mod,args): mod.float()
    def after(mod,args,out): mod.half()
    for layer in engine.backbone.layers:
        hooks.append(layer.register_forward_pre_hook(before))
        hooks.append(layer.register_forward_hook(after, always_call=True))
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        for handle in hooks: handle.remove()
        for layer in engine.backbone.layers: layer.half()
        engine.backbone.norm.half()
        torch.backends.cuda.matmul.allow_tf32=old_tf32
        torch.backends.cudnn.allow_tf32=old_cudnn

def numerical(engine,out,context,questions):
    data={"description":"Repeatability, temperature scaling, and FP32 arithmetic reference for the same FP16-rounded weights."}
    rows={}
    for name,expand in [("default",False),("expanded",True)]:
        rows[name]=[]
        for seed in [0,11,97]:
            torch.manual_seed(seed)
            before=torch.cuda.get_rng_state().clone()
            with routing(expand): result=engine.decide(context,questions)
            result["cuda_rng_unchanged"]=torch.equal(before,torch.cuda.get_rng_state())
            rows[name].append(result)
            print(f"Repeatability {name}, seed {seed}",flush=True)
    data["repeatability"]={k:{"max_probability_spread":spread(v),
        "rng_unchanged":all(x["cuda_rng_unchanged"] for x in v)} for k,v in rows.items()}
    a,b=rows["default"][0],rows["expanded"][0]
    differences=[]
    for k,v in a["answers"].items():
        diff=max(abs(p-b["answers"][k]["probabilities"][o]) for o,p in v["probabilities"].items())
        differences.append((diff,k))
    selected=[k for _,k in sorted(differences,reverse=True)[:4]]
    # Use one original question for a direct deterministic temperature test.
    q={selected[0]:questions[selected[0]]}
    engine.temperature=1.0
    base=engine.decide(context,q)
    try:
        engine.temperature=.5
        half=[engine.decide(context,q) for _ in range(2)]
    finally:
        engine.temperature=1.0
    p=list(base["answers"][selected[0]]["probabilities"].values())
    expected=[x*x/sum(v*v for v in p) for x in p]
    actual=list(half[0]["answers"][selected[0]]["probabilities"].values())
    data["temperature"]={"question":selected[0], "temperature_1":base,
        "temperature_half":half[0], "repeat_spread":spread(half),
        "max_difference_from_normalized_squared_probabilities":max(abs(x-y) for x,y in zip(expected,actual))}
    data["selected_questions"]=selected
    data["original_results"]=rows
    save(out/"numerical.json",data)
    print("FP32 arithmetic reference on four largest-difference questions: "+str(selected),flush=True)
    reference={}
    with fp32_reference(engine):
        for k in selected:
            result=engine.decide(context,{k:questions[k]},mode="independent")
            reference[k]=result["answers"][k]
            print("  FP32 reference complete: "+k,flush=True)
        cached=engine.decide(context,{k:questions[k] for k in selected},mode="parallel")
    data["fp32_reference"]={"answers":reference}
    data["fp32_cached_vs_independent"]=h.compare({"answers":reference},cached)
    data["error_against_fp32"]={}
    for variant in rows:
        data["error_against_fp32"][variant]={
            k: max(abs(p-reference[k]["probabilities"][o]) for o,p in rows[variant][0]["answers"][k]["probabilities"].items())
            for k in selected}
    data["all_parameters_restored_to_fp16"]=all(p.dtype==torch.float16 for p in engine.model.parameters())
    restored=engine.decide(context,questions)
    data["after_reference_vs_before"]=h.compare(a,restored)
    save(out/"numerical.json",data)
    print(json.dumps({k:data[k] for k in ["repeatability","error_against_fp32","fp32_cached_vs_independent","after_reference_vs_before"]}),flush=True)

@contextlib.contextmanager
def cache_fork(enabled):
    from transformers.cache_utils import DynamicCache, DynamicLayer
    original=engine_module.copy
    def fork(cache):
        if type(cache) is not DynamicCache or any(type(layer) is not DynamicLayer for layer in cache.layers):
            raise TypeError("Diagnostic metadata fork only supports ordinary dense DynamicCache layers")
        result=copy.copy(cache)
        result.layers=[copy.copy(layer) for layer in cache.layers]
        # batch_repeat_interleave and update replace the new layers' tensors; they do not mutate the shared prefix.
        return result
    try:
        if enabled: engine_module.copy=types.SimpleNamespace(deepcopy=fork)
        yield
    finally:
        engine_module.copy=original

@contextlib.contextmanager
def variant_config(engine,name):
    old_batch=engine.branch_batch_size
    try:
        if name.startswith("batch"): engine.branch_batch_size=int(name[5:])
        with routing(name=="expanded_kv"), cache_fork(name=="shallow_cache"):
            with sdpa_kernel(SDPBackend.MATH) if name=="math_only" else contextlib.nullcontext():
                yield
    finally: engine.branch_batch_size=old_batch

def native_trace(engine,out,name,context,questions):
    from torch.profiler import profile,ProfilerActivity,record_function
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        return {"status":"unavailable", "reason":"No CUDA Kineto support"}
    for _ in range(2): engine.decide(context,questions)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                 record_shapes=True,profile_memory=False,with_stack=False) as prof:
        with record_function("REQUEST::"+name):
            result=engine.decide(context,questions)
            torch.cuda.synchronize()
    path=out/(name+"-native-trace.json")
    prof.export_chrome_trace(str(path))
    trace=json.loads(path.read_text())
    events=trace["traceEvents"]
    kernels=[e for e in events if e.get("cat")=="kernel" and "dur" in e]
    copies=[e for e in events if e.get("cat") in ("gpu_memcpy","gpu_memset") and "dur" in e]
    grouped=defaultdict(lambda:{"count":0,"total_us":0.,"max_us":0.})
    for e in kernels:
        x=grouped[e["name"]]
        x["count"]+=1; x["total_us"]+=e["dur"]; x["max_us"]=max(x["max_us"],e["dur"])
    intervals=sorted((e["ts"],e["ts"]+e["dur"]) for e in kernels+copies)
    merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
        else: merged.append([start,end])
    request=next(e for e in events if e.get("name")=="REQUEST::"+name and "dur" in e)
    summary={"status":"ok" if kernels else "cuda_api_only","trace":str(path),"kernel_count":len(kernels),
        "kernel_duration_sum_ms":sum(e["dur"] for e in kernels)/1000 if kernels else None,
        "gpu_work_interval_union_ms":sum(b-a for a,b in merged)/1000 if merged else None,
        "request_cpu_range_ms":request["dur"]/1000,
        "memory_transfer_count":len(copies),"memory_transfer_sum_ms":sum(e["dur"] for e in copies)/1000 if copies else None,
        "top_kernels":sorted([dict(name=k,**v) for k,v in grouped.items()],key=lambda x:x["total_us"],reverse=True)[:30],
        "attention_operators":{e.key:e.count for e in prof.key_averages() if "attention" in e.key.lower()},
        "result":result,
        "note":"Trace overhead affects wall time. GPU interval union is elapsed busy intervals, not SM occupancy or bandwidth utilization."}
    save(out/(name+"-native-summary.json"),summary)
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--numerical-only",action="store_true")
    p.add_argument("--repeats",type=int,default=7)
    args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    engine=DecisionEngine.from_pretrained(device="cuda",revision=h.REVISION,local_files_only=True)
    environment={"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,
        "transformers":transformers.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
        "flash_compiled":torch.backends.cuda.is_flash_attention_available(),
        "profiler_activities":[str(x) for x in torch.profiler.supported_activities()],
        "torch_cpu_threads":torch.get_num_threads(),"revision":h.REVISION,
        "engine_sha256":hashlib.sha256((ROOT/"src/sparset_decide/engine.py").read_bytes()).hexdigest()}
    save(out/"environment.json",environment)
    refund,rq=load_request(ROOT/"examples/decision_engine/refund.json")
    support,sq=load_request(ROOT/"outputs/decision-engine/replica-presets/support_triage.request.json")
    if args.numerical_only:
        numerical(engine,out,support["context"],sq)
        return
    first=next(iter(rq))
    long=("Historical background: an earlier order was delivered successfully and its invoice was archived.\n"*55
          +"\nCURRENT CUSTOMER MESSAGE:\n"+refund["context"])
    cases=[("refund_1",refund["context"],{first:rq[first]}),
           ("refund_4",refund["context"],rq),("support_28",support["context"],sq),("long_context_4",long,rq)]
    report={"environment":environment,"cases":{}}
    for name,context,questions in cases:
        variants=["auto","math_only","expanded_kv"]
        if name=="support_28": variants+=["batch4","batch16","batch28","sorted_batch8","shallow_cache"]
        rows={v:[] for v in variants}; errors={}
        seq=engine.prepare(context,questions)
        sorted_q={k:questions[k] for _,k in sorted(zip(map(len,seq),questions))}
        for v in variants:
            try:
                with variant_config(engine,v):
                    for _ in range(2): engine.decide(context,sorted_q if v=="sorted_batch8" else questions)
            except Exception as exc:
                errors[v]=repr(exc); torch.cuda.empty_cache()
        for i in range(args.repeats):
            order=variants[i%len(variants):]+variants[:i%len(variants)]
            for v in order:
                if v in errors: continue
                try:
                    with variant_config(engine,v):
                        torch.cuda.reset_peak_memory_stats()
                        result=engine.decide(context,sorted_q if v=="sorted_batch8" else questions)
                        result["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
                        rows[v].append(result)
                except Exception as exc:
                    errors[v]=repr(exc); torch.cuda.empty_cache()
            print(f"{name}: repetition {i+1}/{args.repeats}",flush=True)
        case={"errors":errors,"summary":{v:h.timing(x) for v,x in rows.items() if x},
              "repeatability":{v:spread(x) for v,x in rows.items() if x},
              "parity_vs_auto":{v:h.compare(rows["auto"][0],x[0]) for v,x in rows.items() if x},
              "peak_allocated_bytes":{v:max(z["peak_allocated_bytes"] for z in x) for v,x in rows.items() if x},
              "results":rows}
        report["cases"][name]=case; save(out/"performance.json",report)
        print(json.dumps({"case":name,"median_ms":{v:x["median_ms"] for v,x in case["summary"].items()},
                          "repeatability":case["repeatability"],"errors":errors}),flush=True)
    # Native tracing after every unprofiled timing run has finished.
    report["native_traces"]={}
    for name,context,questions in cases:
        print("Native GPU trace: "+name,flush=True)
        report["native_traces"][name]=native_trace(engine,out,name,context,questions)
        save(out/"performance.json",report)
    # CPU launch overhead check with inputs and weights already resident.
    print("Complete: "+str(out),flush=True)
if __name__=="__main__": main()
