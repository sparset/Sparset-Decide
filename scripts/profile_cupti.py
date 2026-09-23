"""Direct CUPTI 12.4 activity diagnostic, bypassing Kineto trace filtering."""
import faulthandler
faulthandler.dump_traceback_later(120, repeat=False)
print("CUPTI diagnostic: importing dependencies",flush=True)
import ctypes as C
import json,sys,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import profile_engine as h
import torch

class KernelPrefix(C.Structure):
    _pack_=1
    _fields_=[("kind",C.c_uint32),("cacheConfig",C.c_uint8),("sharedMemoryConfig",C.c_uint8),
        ("registersPerThread",C.c_uint16),("partitionRequested",C.c_uint32),("partitionExecuted",C.c_uint32),
        ("start",C.c_uint64),("end",C.c_uint64),("completed",C.c_uint64),
        ("device",C.c_uint32),("context",C.c_uint32),("stream",C.c_uint32),
        ("gridX",C.c_int32),("gridY",C.c_int32),("gridZ",C.c_int32),
        ("blockX",C.c_int32),("blockY",C.c_int32),("blockZ",C.c_int32),
        ("staticSharedMemory",C.c_int32),("dynamicSharedMemory",C.c_int32),
        ("localMemoryPerThread",C.c_uint32),("localMemoryTotal",C.c_uint32),("correlation",C.c_uint32),
        ("gridId",C.c_int64),("name",C.c_void_p)]

class Cupti:
    def __init__(self):
        lib=next(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"))
        self.lib=C.CDLL(str(lib))
        self.lib.cuptiGetResultString.argtypes=[C.c_int,C.POINTER(C.c_char_p)]
        self.lib.cuptiGetVersion.argtypes=[C.POINTER(C.c_uint32)]
        version=C.c_uint32(); self.check(self.lib.cuptiGetVersion(C.byref(version)))
        if version.value!=22: raise RuntimeError("Only the verified CUPTI API 22 (CUDA 12.4) record layout is supported")
        self.records=[]; self.errors=[]; self.buffers={}; self.kinds=collections.Counter()
        self.buffer_requests=0; self.buffer_completions=0; self.dropped=0
        U8=C.POINTER(C.c_uint8)
        Request=C.CFUNCTYPE(None,C.POINTER(U8),C.POINTER(C.c_size_t),C.POINTER(C.c_size_t))
        Complete=C.CFUNCTYPE(None,C.c_void_p,C.c_uint32,U8,C.c_size_t,C.c_size_t)
        self.lib.cuptiActivityGetNextRecord.argtypes=[U8,C.c_size_t,C.POINTER(C.c_void_p)]
        self.lib.cuptiActivityGetNumDroppedRecords.argtypes=[C.c_void_p,C.c_uint32,C.POINTER(C.c_size_t)]
        self.lib.cuptiActivityRegisterCallbacks.argtypes=[Request,Complete]
        self.lib.cuptiActivityEnable.argtypes=[C.c_int]
        self.lib.cuptiActivityDisable.argtypes=[C.c_int]
        self.lib.cuptiActivityFlushAll.argtypes=[C.c_uint32]
        def requested(buffer,size,max_records):
            allocation=C.create_string_buffer(4*1024*1024)
            address=C.addressof(allocation); self.buffers[address]=allocation
            buffer[0]=C.cast(address,U8); size[0]=len(allocation); max_records[0]=0
            self.buffer_requests+=1
        def completed(context,stream,buffer,size,valid):
            self.buffer_completions+=1
            try:
                pointer=C.c_void_p()
                while valid:
                    status=self.lib.cuptiActivityGetNextRecord(buffer,valid,C.byref(pointer))
                    if status:
                        if "MAX_LIMIT_REACHED" not in self.message(status): self.errors.append(self.message(status))
                        break
                    kind=C.cast(pointer,C.POINTER(C.c_uint32))[0]; self.kinds[kind]+=1
                    if kind not in (3,10): continue
                    record=KernelPrefix.from_address(pointer.value)
                    self.records.append({"name":C.string_at(record.name).decode(errors="replace") if record.name else "",
                        "start_ns":record.start,"end_ns":record.end,
                        "duration_us":(record.end-record.start)/1000 if record.end>=record.start and record.start else None,
                        "stream":record.stream,"correlation":record.correlation,"registers_per_thread":record.registersPerThread,
                        "grid":[record.gridX,record.gridY,record.gridZ],
                        "block":[record.blockX,record.blockY,record.blockZ]})
                dropped=C.c_size_t()
                status=self.lib.cuptiActivityGetNumDroppedRecords(context,stream,C.byref(dropped))
                if not status: self.dropped+=dropped.value
            except Exception as exc: self.errors.append(repr(exc))
            finally: self.buffers.pop(C.cast(buffer,C.c_void_p).value,None)
        self.request=Request(requested); self.complete=Complete(completed)
        self.check(self.lib.cuptiActivityRegisterCallbacks(self.request,self.complete))
    def message(self,result):
        s=C.c_char_p(); self.lib.cuptiGetResultString(result,C.byref(s))
        return s.value.decode() if s.value else str(result)
    def check(self,result):
        if result: raise RuntimeError(self.message(result))
    def start(self): self.check(self.lib.cuptiActivityEnable(10))
    def stop(self):
        torch.cuda.synchronize()
        self.check(self.lib.cuptiActivityFlushAll(1))
        self.check(self.lib.cuptiActivityDisable(10))
        self.check(self.lib.cuptiActivityFlushAll(1))
    def report(self):
        return {"kernel_records":self.records,"record_kinds":dict(self.kinds),"dropped":self.dropped,
            "errors":self.errors,"buffer_requests":self.buffer_requests,"buffer_completions":self.buffer_completions}

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model")
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    try:
        if args.model:
            from sparset_decide.engine import DecisionEngine
            from sparset_decide.__main__ import load_request
            engine=DecisionEngine.from_pretrained(args.model,device="cuda",local_files_only=True)
            data,questions=load_request(ROOT/"examples/decision_engine/refund.json")
            for _ in range(3): engine.decide(data["context"],questions)
            print("CUPTI diagnostic: registering and enabling activity",flush=True)
            cupti=Cupti(); cupti.start()
            start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); begin=time.perf_counter()
            result=engine.decide(data["context"],questions)
            stop.record(); stop.synchronize()
            wall=(time.perf_counter()-begin)*1000
            cupti.stop()
            report=cupti.report(); report["result"]=result
            report["profiled_wall_ms"]=wall; report["profiled_cuda_event_ms"]=start.elapsed_time(stop)
        else:
            print("CUPTI diagnostic: creating CUDA tensors",flush=True)
            a=torch.ones(512,512,device="cuda"); b=torch.ones_like(a)
            for _ in range(3): c=a@b
            torch.cuda.synchronize()
            print("CUPTI diagnostic: registering and enabling activity",flush=True)
            cupti=Cupti(); cupti.start()
            for _ in range(10): c=a@b
            print("CUPTI diagnostic: flushing activity",flush=True)
            cupti.stop(); report=cupti.report()
        report["status"]="collected" if report["kernel_records"] else "no_gpu_records"
    except Exception as exc:
        report={"status":"failed","error":repr(exc)}
    h.save(args.output/"summary.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("kernel_records","result")},indent=2),flush=True)
    print("Native kernel records:",len(report.get("kernel_records",[])),flush=True)
if __name__=="__main__": main()
