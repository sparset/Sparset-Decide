"""Bounded, exact-shape CUDA graph replay for immutable inference models.

Graph-owned buffers are private to an engine; decide serializes access. New token
values, masks and prefix KV tensors are copied on every replay. Unsupported
cache implementations and failed/oversized captures use the eager path.
"""
import copy
from collections import OrderedDict
import torch


def fork_dense_cache(cache):
    from transformers.cache_utils import DynamicCache, DynamicLayer
    if type(cache) is not DynamicCache or getattr(cache, 'offloading', False) or any(type(x) is not DynamicLayer for x in cache.layers):
        raise TypeError('Graph replay requires ordinary dense DynamicCache layers')
    result=copy.copy(cache)
    result.layers=[copy.copy(x) for x in cache.layers]
    return result


class GraphRunner:
    def __init__(self, engine, max_graphs=4, budget_mb=192):
        self.engine=engine; self.max_graphs=max_graphs; self.budget=budget_mb*1024**2
        self.entries=OrderedDict(); self.failed=OrderedDict()
        self.stats={'captures':0,'replays':0,'fallbacks':0,'evictions':0}
        self.last_fallback=None

    def clear(self):
        self.entries.clear();self.failed.clear()

    def _reject(self,key,reason):
        self.failed[key]=reason;self.failed.move_to_end(key)
        while len(self.failed)>64:self.failed.popitem(last=False)
        self.last_fallback=reason;self.stats['fallbacks']+=1

    def run(self, rows, cache, prefix_length, use_cache):
        e=self.engine
        if e.device.type!='cuda' or e.model.config._attn_implementation!='sdpa':
            self.stats['fallbacks']+=1;self.last_fallback='requires CUDA and SDPA';return None
        try:
            from transformers.cache_utils import DynamicCache, DynamicLayer
        except ImportError:
            self.stats['fallbacks']+=1;self.last_fallback='unsupported Transformers cache API';return None
        if cache is not None:
            try:fork_dense_cache(cache)
            except (TypeError,ImportError):
                self.stats['fallbacks']+=1;self.last_fallback='unsupported cache';return None
        lengths=[len(x)for x in rows];batch=len(rows);width=max(lengths)
        unmasked=prefix_length==0 and all(x==width for x in lengths)
        key=(batch,width,prefix_length,use_cache,unmasked,tuple(e.fused_kernels))
        if key in self.failed:
            self.stats['fallbacks']+=1;self.last_fallback=self.failed[key];return None
        pad=e.tokenizer.pad_token_id
        if pad is None:pad=e.tokenizer.eos_token_id
        ids=torch.tensor([x+[pad]*(width-len(x))for x in rows],device=e.device)
        positions=(torch.arange(width,device=e.device)+prefix_length)[None,:].expand(batch,-1).contiguous()
        indices=torch.tensor(lengths,device=e.device)-1
        mask=None
        if not unmasked:
            keys=torch.arange(prefix_length+width,device=e.device)
            valid=keys[None,:]<(torch.tensor(lengths,device=e.device)+prefix_length)[:,None]
            allowed=(keys[None,:]<=positions[0,:,None])[None,None,:,:] & valid[:,None,None,:]
            mask=torch.zeros(allowed.shape,dtype=e.head.weight.dtype,device=e.device).masked_fill_(~allowed,torch.finfo(e.head.weight.dtype).min)
        entry=self.entries.get(key)
        if entry is None and cache is not None:
            # Input prefix copies plus output KV and attention/MLP scratch can
            # exceed physical VRAM before a capture finishes. Reject expensive
            # shapes before allocating them on small-memory configurations.
            kv_bytes=sum(x.keys.numel()*x.keys.element_size()+x.values.numel()*x.values.element_size() for x in cache.layers)
            if 4*kv_bytes + sum(x['bytes'] for x in self.entries.values()) > self.budget:
                self._reject(key,'prefix KV and scratch estimate exceeds graph memory budget');return None
        if entry is None:
            while len(self.entries)>=self.max_graphs:
                self.entries.popitem(last=False);self.stats['evictions']+=1
            # Separate private pools allow entries to be replayed in arbitrary order.
            # Evict before capture when little device memory remains.
            free,_=torch.cuda.mem_get_info(e.device)
            if free+torch.cuda.memory_reserved(e.device)-torch.cuda.memory_allocated(e.device)<256*1024**2:
                self._reject(key,'insufficient free memory for another graph');return None
            before=torch.cuda.memory_allocated(e.device)
            try:
                entry=self._capture(ids,positions,indices,mask,cache,use_cache)
                entry['bytes']=max(0,torch.cuda.memory_allocated(e.device)-before)
                if entry['bytes']>self.budget:
                    entry=None;torch.cuda.empty_cache();self._reject(key,'capture exceeds graph memory budget');return None
                if sum(x['bytes'] for x in self.entries.values())+entry['bytes'] > self.budget:
                    entry=None;torch.cuda.empty_cache();self._reject(key,'total graph memory budget reached');return None
                self.entries[key]=entry;self.stats['captures']+=1
            except (RuntimeError,TypeError,AttributeError,NotImplementedError) as exc:
                entry=None;torch.cuda.empty_cache();self._reject(key,type(exc).__name__+': '+str(exc)[:240]);return None
        else:
            self.entries.move_to_end(key)
        entry['ids'].copy_(ids);entry['positions'].copy_(positions);entry['indices'].copy_(indices)
        if mask is not None:entry['mask'].copy_(mask)
        if cache is not None:
            for dst,src in zip(entry['cache'].layers,cache.layers):
                dst.keys.copy_(src.keys);dst.values.copy_(src.values)
        entry['graph'].replay();self.stats['replays']+=1
        hidden,output_cache=entry['output']
        return hidden, None if output_cache is None else fork_dense_cache(output_cache)

    def _capture(self,ids,positions,indices,mask,cache,use_cache):
        e=self.engine
        ids=ids.clone();positions=positions.clone();indices=indices.clone()
        mask=None if mask is None else mask.clone()
        if cache is not None:
            cache=fork_dense_cache(cache)
            for layer in cache.layers:
                layer.keys=layer.keys.clone();layer.values=layer.values.clone()
        batch_indices=torch.arange(ids.shape[0],device=e.device)
        def core():
            current=None if cache is None else fork_dense_cache(cache)
            out=e.backbone(input_ids=ids,position_ids=positions,attention_mask=mask,
                           past_key_values=current,use_cache=use_cache,return_dict=True)
            return out.last_hidden_state[batch_indices,indices],out.past_key_values
        # cuBLAS retains a workspace per stream. Reusing one model-scoped stream
        # prevents persistent workspace growth when request shapes change.
        if getattr(e._fusion, 'capture_stream', None) is None:
            e._fusion.capture_stream=torch.cuda.Stream(device=e.device)
        stream=e._fusion.capture_stream
        stream.wait_stream(torch.cuda.current_stream(e.device))
        with torch.cuda.stream(stream):
            for _ in range(3):output=core()
        torch.cuda.current_stream(e.device).wait_stream(stream)
        del output
        torch.cuda.synchronize(e.device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):output=core()
        if output[1] is not None:
            fork_dense_cache(output[1])  # Validate before admitting the captured output cache.
        return dict(ids=ids,positions=positions,indices=indices,batch_indices=batch_indices,mask=mask,cache=cache,graph=graph,output=output)