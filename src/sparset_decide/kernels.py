"""Optional inference-only Triton kernels. Imported only when explicitly enabled.

Preserve intermediate FP16/BF16 rounding used by the reference operations.
No training/backward implementation. No global Transformers monkey patches.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _rms(X,W,Y,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+i,i<N,other=0).to(tl.float32)
    variance=tl.sum(x*x,0)/N
    norm=(x*tl.rsqrt(variance+EPS)).to(Y.dtype.element_ty).to(tl.float32)
    w=tl.load(W+i,i<N,other=0).to(tl.float32)
    tl.store(Y+row*N+i,norm*w,i<N)

@triton.jit
def _swiglu(G,U,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(G+i,i<N,other=0).to(tl.float32)
    u=tl.load(U+i,i<N,other=0).to(tl.float32)
    activated=(x/(1+tl.exp(-x))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,activated*u,i<N)

@triton.jit
def _rotate_one(X,C,S,Y,BASE,N:tl.constexpr,H:tl.constexpr,T:tl.constexpr,D:tl.constexpr,
                XB:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,CB:tl.constexpr,CT:tl.constexpr,BLOCK:tl.constexpr):
    i=BASE+tl.arange(0,BLOCK);d=i%D;t=(i//D)%T;h=(i//(D*T))%H;b=i//(D*T*H)
    pos=b*XB+h*XH+t*XT
    x=tl.load(X+pos+d,i<N,other=0).to(tl.float32)
    rotated=tl.load(X+pos+(d+D//2)%D,i<N,other=0).to(tl.float32)
    rotated=tl.where(d<D//2,-rotated,rotated)
    c=tl.load(C+b*CB+t*CT+d,i<N,other=0).to(tl.float32)
    s=tl.load(S+b*CB+t*CT+d,i<N,other=0).to(tl.float32)
    a=(x*c).to(Y.dtype.element_ty).to(tl.float32)
    z=(rotated*s).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+i,a+z,i<N)

@triton.jit
def _rope(Q,K,C,S,OQ,OK,NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,T:tl.constexpr,D:tl.constexpr,
          QB:tl.constexpr,QH:tl.constexpr,QT:tl.constexpr,KB:tl.constexpr,KH:tl.constexpr,KT:tl.constexpr,
          CB:tl.constexpr,CT:tl.constexpr,QBLOCKS:tl.constexpr,BLOCK:tl.constexpr):
    p=tl.program_id(0)
    if p<QBLOCKS:
        _rotate_one(Q,C,S,OQ,p*BLOCK,NQ,HQ,T,D,QB,QH,QT,CB,CT,BLOCK)
    else:
        _rotate_one(K,C,S,OK,(p-QBLOCKS)*BLOCK,NK,HK,T,D,KB,KH,KT,CB,CT,BLOCK)

def eligible(x):
    return x.is_cuda and x.dtype in (torch.float16,torch.bfloat16,torch.float32) and not torch.is_grad_enabled()

def rmsnorm(x,weight,eps):
    if not eligible(x) or not x.is_contiguous() or x.shape[-1]>16384:
        y=x.float();return weight*(y*torch.rsqrt(y.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
    out=torch.empty_like(x);n=x.shape[-1]
    _rms[(x.numel()//n,)](x,weight,out,n,eps,triton.next_power_of_2(n),enable_fp_fusion=False)
    return out

def swiglu(gate,up):
    if not eligible(gate) or not gate.is_contiguous() or not up.is_contiguous():
        return torch.nn.functional.silu(gate)*up
    out=torch.empty_like(gate);n=gate.numel()
    _swiglu[(triton.cdiv(n,512),)](gate,up,out,n,512,enable_fp_fusion=False)
    return out

def rope(q,k,cos,sin,unsqueeze_dim=1):
    if (not eligible(q) or unsqueeze_dim!=1 or q.shape[-1]%2 or q.stride(-1)!=1 or k.stride(-1)!=1
        or cos.ndim!=3 or not cos.is_contiguous() or not sin.is_contiguous() or cos.shape!=sin.shape):
        c=cos.unsqueeze(unsqueeze_dim);s=sin.unsqueeze(unsqueeze_dim)
        def rotate(x):return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),-1)
        return q*c+rotate(q)*s,k*c+rotate(k)*s
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype);ok=torch.empty(k.shape,device=k.device,dtype=k.dtype)
    nq,nk=q.numel(),k.numel();block=512;qb=triton.cdiv(nq,block)
    _rope[(qb+triton.cdiv(nk,block),)](q,k,cos,sin,oq,ok,nq,nk,q.shape[1],k.shape[1],q.shape[2],q.shape[3],
        *q.stride()[:3],*k.stride()[:3],cos.stride(0)if cos.shape[0]>1 else 0,cos.stride(1),qb,block,enable_fp_fusion=False)
    return oq,ok
@triton.jit
def _residual_rms(X,R,W,S,Y,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+i,i<N,other=0).to(tl.float32)
    r=tl.load(R+row*N+i,i<N,other=0).to(tl.float32)
    added=(x+r).to(S.dtype.element_ty)
    tl.store(S+row*N+i,added,i<N)
    f=added.to(tl.float32);variance=tl.sum(f*f,0)/N
    norm=(f*tl.rsqrt(variance+EPS)).to(Y.dtype.element_ty).to(tl.float32)
    w=tl.load(W+i,i<N,other=0).to(tl.float32)
    tl.store(Y+row*N+i,norm*w,i<N)

def residual_rmsnorm(x,residual,weight,eps):
    if not eligible(x) or not x.is_contiguous() or not residual.is_contiguous() or x.shape[-1]>16384:
        added=x+residual
        return added,rmsnorm(added,weight,eps)
    added=torch.empty_like(x);out=torch.empty_like(x);n=x.shape[-1]
    _residual_rms[(x.numel()//n,)](x,residual,weight,added,out,n,eps,triton.next_power_of_2(n),enable_fp_fusion=False)
    return added,out
