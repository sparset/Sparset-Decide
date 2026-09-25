"""Scoped model adapters for the validated Transformers Qwen2 implementation."""
import types
import threading
from contextlib import contextmanager

class FusionManager:
    def __init__(self,model):
        self.model=model;self.lock=threading.RLock();self.active=frozenset();self.installed=False;self.branch_state=None

    def install(self):
        if self.installed:return
        import transformers
        if transformers.__version__!='5.5.4' or self.model.config.model_type!='qwen2':
            raise ValueError('Custom fusion currently requires dense Qwen2 and transformers==5.5.4; other models use the ordinary engine')
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm,Qwen2MLP,Qwen2Attention,Qwen2DecoderLayer
        from . import kernels
        if self.model.config.hidden_act!='silu':raise ValueError('SwiGLU fusion requires silu activation')
        # Validate the complete model before modifying any forward methods.
        for module in self.model.modules():
            if type(module) is Qwen2Attention and 'apply_rotary_pos_emb' not in module.forward.__func__.__globals__:
                raise ValueError('Unsupported wrapped attention implementation')
        manager=self
        for module in self.model.modules():
            original=module.forward
            if type(module)is Qwen2RMSNorm:
                def norm(this,x,_original=original):
                    return kernels.rmsnorm(x,this.weight,this.variance_epsilon)if 'rmsnorm'in manager.active else _original(x)
                module.forward=types.MethodType(norm,module)
            elif type(module)is Qwen2MLP:
                def mlp(this,x,_original=original):
                    if 'swiglu'not in manager.active:return _original(x)
                    return this.down_proj(kernels.swiglu(this.gate_proj(x),this.up_proj(x)))
                module.forward=types.MethodType(mlp,module)
            elif type(module)is Qwen2DecoderLayer:
                def decoder(this,hidden_states,attention_mask=None,position_ids=None,past_key_values=None,
                            use_cache=False,position_embeddings=None,_original=original,**kwargs):
                    if 'residual_norm'not in manager.active:
                        return _original(hidden_states,attention_mask=attention_mask,position_ids=position_ids,
                                         past_key_values=past_key_values,use_cache=use_cache,
                                         position_embeddings=position_embeddings,**kwargs)
                    residual=hidden_states
                    x=this.input_layernorm(hidden_states)
                    x,_=this.self_attn(hidden_states=x,attention_mask=attention_mask,position_ids=position_ids,
                                      past_key_values=past_key_values,use_cache=use_cache,
                                      position_embeddings=position_embeddings,**kwargs)
                    residual,x=kernels.residual_rmsnorm(x,residual,this.post_attention_layernorm.weight,
                                                       this.post_attention_layernorm.variance_epsilon)
                    return residual+this.mlp(x)
                module.forward=types.MethodType(decoder,module)
            elif type(module)is Qwen2Attention:
                # Clone the function's namespace, not the library module namespace.
                # This preserves upstream attention/cache behavior while replacing
                # only RoPE inside this model instance.
                fn=original.__func__
                if 'apply_rotary_pos_emb'not in fn.__globals__:
                    raise ValueError('Unsupported wrapped attention implementation')
                original_rope=fn.__globals__['apply_rotary_pos_emb']
                def rotate(*args,_original=original_rope,**kwargs):
                    return kernels.rope(*args,**kwargs)if 'rope'in manager.active else _original(*args,**kwargs)
                glob=dict(fn.__globals__);glob['apply_rotary_pos_emb']=rotate
                replacement=types.FunctionType(fn.__code__,glob,fn.__name__,fn.__defaults__,fn.__closure__)
                replacement.__kwdefaults__=fn.__kwdefaults__
                def attend(this,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,
                           _replacement=replacement,_rotate=rotate,**kwargs):
                    if manager.branch_state is not None:
                        from .shared_attention import forward
                        return forward(this,hidden_states,position_embeddings,manager.branch_state,_rotate)
                    return _replacement(this,hidden_states,position_embeddings,attention_mask,
                                        past_key_values=past_key_values,**kwargs)
                module.forward=types.MethodType(attend,module)
        self.installed=True

    @contextmanager
    def use(self,names):
        with self.lock:
            if names:self.install()
            old=self.active;self.active=frozenset(names)
            try:yield
            finally:self.active=old
    @contextmanager
    def shared_branches(self,cache,lengths,specialize_short):
        with self.lock:
            self.install()
            previous=self.branch_state
            self.branch_state={'cache':cache,'lengths':lengths,'specialize_short':specialize_short}
            try:yield
            finally:self.branch_state=previous
