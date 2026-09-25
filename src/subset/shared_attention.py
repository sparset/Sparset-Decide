"""Inference-only attention over one shared prefix and private branch suffixes.

No prefix repetition/concatenation. No approximations or context truncation.
The engine falls back to ordinary Transformers outside the validated envelope.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _attend(Q,K,V,PK,PV,L,O,T,P,
            HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
            QB,QH,QT,
            KB,KH,KT,
            VB,VH,VT,
            PKH,PKT,PVH,PVT,
            SCALE:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    block=tl.program_id(0);bh=tl.program_id(1);b=bh//HQ;h=bh%HQ;kh=h//(HQ//HK)
    qi=block*BQ+tl.arange(0,BQ);di=tl.arange(0,BD);ki=tl.arange(0,BK)
    q=tl.load(Q+b*QB+h*QH+qi[:,None]*QT+di[None,:],(qi[:,None]<T)&(di[None,:]<D),other=0)
    m=tl.full((BQ,),float('-inf'),tl.float32);den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,BD),tl.float32)
    for start in range(tl.cdiv(P,BK)):
        pos=start*BK+ki
        k=tl.load(PK+kh*PKH+pos[None,:]*PKT+di[:,None],(pos[None,:]<P)&(di[:,None]<D),other=0)
        v=tl.load(PV+kh*PVH+pos[:,None]*PVT+di[None,:],(pos[:,None]<P)&(di[None,:]<D),other=0)
        scores=tl.dot(q,k)*SCALE
        scores=tl.where(pos[None,:]<P,scores,float('-inf'))
        new_m=tl.maximum(m,tl.max(scores,1));alpha=tl.exp(m-new_m)
        prob=tl.exp(scores-new_m[:,None]);den=den*alpha+tl.sum(prob,1)
        acc=acc*alpha[:,None]+tl.dot(prob.to(q.dtype),v);m=new_m
    length=tl.load(L+b)
    for start in range(tl.cdiv(T,BK)):
        pos=start*BK+ki
        k=tl.load(K+b*KB+kh*KH+pos[None,:]*KT+di[:,None],(pos[None,:]<T)&(di[:,None]<D),other=0)
        v=tl.load(V+b*VB+kh*VH+pos[:,None]*VT+di[None,:],(pos[:,None]<T)&(di[None,:]<D),other=0)
        scores=tl.dot(q,k)*SCALE
        scores=tl.where((pos[None,:]<length)&(pos[None,:]<=qi[:,None]),scores,float('-inf'))
        new_m=tl.maximum(m,tl.max(scores,1));alpha=tl.exp(m-new_m)
        prob=tl.exp(scores-new_m[:,None]);den=den*alpha+tl.sum(prob,1)
        acc=acc*alpha[:,None]+tl.dot(prob.to(q.dtype),v);m=new_m
    result=acc/den[:,None]
    tl.store(O+((b*HQ+h)*T+qi[:,None])*D+di[None,:],result,(qi[:,None]<T)&(di[None,:]<D))

def attention(q,k,v,prefix_k,prefix_v,lengths,scale,specialize_short=True):
    b,h,t,d=q.shape;p=prefix_k.shape[-2]
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    tile=16 if specialize_short and t<=16 else 32
    _attend[(triton.cdiv(t,tile),b*h)](q,k,v,prefix_k,prefix_v,lengths,out,t,p,
        h,k.shape[1],d,*q.stride()[:3],*k.stride()[:3],*v.stride()[:3],
        prefix_k.stride(1),prefix_k.stride(2),prefix_v.stride(1),prefix_v.stride(2),
        scale,tile,64,triton.next_power_of_2(d),num_warps=4)
    return out

def forward(module,hidden_states,position_embeddings,state,rotate):
    shape=hidden_states.shape[:-1];view=(*shape,-1,module.head_dim)
    q=module.q_proj(hidden_states).view(view).transpose(1,2)
    k=module.k_proj(hidden_states).view(view).transpose(1,2)
    v=module.v_proj(hidden_states).view(view).transpose(1,2)
    q,k=rotate(q,k,*position_embeddings)
    layer=state['cache'].layers[module.layer_idx]
    value=attention(q,k,v,layer.keys,layer.values,state['lengths'],module.scaling,state['specialize_short'])
    value=value.transpose(1,2).reshape(*shape,-1).contiguous()
    return module.o_proj(value),None
