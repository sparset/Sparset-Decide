"""Verify the promoted default on every frozen case with production optimizations."""
import json,gc,hashlib,time
from run_formal_benchmark import ROOT,MODEL,DATA,DecisionEngine,torch,question_from_dict,validate,append,existing
OUT=ROOT/"outputs/decision-engine/prompt-bias-20260918"
def main():
    e=DecisionEngine.from_pretrained(MODEL,device="cuda",local_files_only=True,cuda_graphs=True,fused_kernels=("rmsnorm","swiglu","rope"))
    assert (e.prompt_format,e.answer_encoding)==("delimited","letters")
    e.warmup()
    allcases=[("core",c) for c in DATA["cases"]]+[("robustness",c) for c in DATA["robustness_cases"]]+[("additional",c) for c in json.loads((ROOT/"outputs/decision-engine/accuracy-diagnosis-20260918/fresh-validation.json").read_text())["cases"]]+[("new_holdout",c) for c in json.loads((OUT/"new-holdout.json").read_text())["cases"]]
    reference={r["case_id"]:r for r in existing(OUT/"validation.jsonl") if r["method"]=="delimited"}
    done={r["case_id"] for r in existing(OUT/"production.jsonl")}
    for i,(suite,c) in enumerate(allcases):
        if c["id"] in done:continue
        qs={k:question_from_dict(v) for k,v in c["questions"].items()}
        torch.cuda.synchronize();start=time.perf_counter();native=e.decide(c["context"],qs);torch.cuda.synchronize()
        answers={k:{"choice":max(a["probabilities"],key=a["probabilities"].get),"probabilities":a["probabilities"]} for k,a in native["answers"].items()}
        validate(answers,qs);ref=reference[c["id"]]["result"]["answers"]
        delta=max(abs(p-ref[k]["probabilities"][label]) for k,a in answers.items() for label,p in a["probabilities"].items())
        row={"case_id":c["id"],"suite":suite,"group":c["group"],"expected":c["expected"],"correct":all(answers[k]["choice"]==v for k,v in c["expected"].items()),"wall_ms":(time.perf_counter()-start)*1000,"max_probability_delta_from_eager":delta,"choice_changed_from_eager":any(a["choice"]!=ref[k]["choice"] for k,a in answers.items()),"result":{"status":"valid","answers":answers,"native":native}}
        append(OUT/"production.jsonl",row)
        if (i+1)%25==0:print("PRODUCTION",i+1,"/",len(allcases),flush=True)
    (OUT/"production-source-manifest.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"src/sparset_decide").glob("*.py")},indent=2),encoding="utf-8")
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
