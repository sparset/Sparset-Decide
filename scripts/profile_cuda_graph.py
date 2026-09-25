"""Measure CUDA-graph replay of a fixed-shape decision core; not an end-to-end engine change."""
import argparse,sys,time,statistics,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import profile_engine as h
import torch
from subset.engine import DecisionEngine
from subset.__main__ import load_request

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen2.5-1.5B-Instruct")
    args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    engine=DecisionEngine.from_pretrained(args.model,device="cuda",revision=h.REVISION,local_files_only=True)
    data,qs=load_request(ROOT/"examples/decision_engine/refund.json")
    key="refund_requested"; questions={key:qs[key]}
    rows=engine.prepare(data["context"],questions)
    ids=torch.tensor(rows,device="cuda")
    positions=torch.arange(ids.shape[1],device="cuda")[None,:]
    count=len(qs[key].options)
    weights=engine.head.weight.index_select(0,engine.answer_ids[:count]).float()
    bias=None if engine.head.bias is None else engine.head.bias.index_select(0,engine.answer_ids[:count]).float()
    @torch.inference_mode()
    def core():
        output=engine.backbone(input_ids=ids,position_ids=positions,attention_mask=None,
                               use_cache=False,return_dict=True)
        return torch.nn.functional.linear(output.last_hidden_state[:,-1,:].float(),weights,bias).softmax(dim=-1)
    result={"scope":"Fixed-shape, single-question GPU core with prepared input tensors and selected output-head weights. Tokenization, input copies, head-row preparation and JSON output excluded.",
            "input_tokens":ids.shape[1],"model_revision":h.REVISION}
    try:
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5): core()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        expected=core().clone()
        graph=torch.cuda.CUDAGraph()
        print("Capturing fixed-shape CUDA graph",flush=True)
        with torch.cuda.graph(graph):
            captured=core()
        graph.replay(); torch.cuda.synchronize()
        result["max_probability_difference"]=float((expected-captured).abs().max().item())
        result["cuda_graph_pool_bytes_after_capture"]=torch.cuda.memory_allocated()
        times={"eager":[],"graph":[]}
        for i in range(20):
            for mode in (("eager","graph") if i%2==0 else ("graph","eager")):
                torch.cuda.synchronize()
                start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin=time.perf_counter(); start.record()
                if mode=="eager": current=core()
                else: graph.replay()
                stop.record(); stop.synchronize()
                times[mode].append({"wall_ms":(time.perf_counter()-begin)*1000,
                                    "cuda_event_ms":start.elapsed_time(stop)})
        result["samples"]=times
        result["summary"]={k:{metric:statistics.median(r[metric] for r in v)
                             for metric in ("wall_ms","cuda_event_ms")} for k,v in times.items()}
        alternative=engine.prepare(data["context"].replace("refund","return"),questions)
        if len(alternative[0])==ids.shape[1]:
            ids.copy_(torch.tensor(alternative,device="cuda"))
            expected_alt=core().clone()
            graph.replay(); torch.cuda.synchronize()
            result["changed_input_max_probability_difference"]=float((expected_alt-captured).abs().max().item())
            result["changed_input_probabilities"]=captured.cpu().tolist()
        result["status"]="ok"
    except Exception as exc:
        result["status"]="failed"; result["error"]=repr(exc)
    h.save(out/"summary.json",result)
    print(json.dumps(result.get("summary",result),indent=2),flush=True)
if __name__=="__main__":main()
