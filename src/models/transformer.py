"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional
from contextlib import contextmanager
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache


@dataclass
class TransformerConfig:

    max_blocks: int

    num_layers: int
    num_heads: int
    embed_dim: int
    tokens_per_block: int
    
    # model_name: str = "stabilityai/stablelm-3b-4e1t"
    # https://huggingface.co/PY007/TinyLlama-1.1B-intermediate-step-715k-1.5T
    model_name: str = "PY007/TinyLlama-1.1B-intermediate-step-715k-1.5T"
    dropout: float = 0
    rank: int = 32

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = load_pretrained_model(config)
        self.ln_f = nn.Linear(self.model.config.vocab_size, config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == self.config.num_layers
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(
                inputs_embeds=sequences,
                return_dict=True,
                output_hidden_states=True,
            )
        x = outputs.logits
        x = self.ln_f(x)
        
        # fake it, since it's used to keep track of steps
        if past_keys_values is not None:
            k_size = past_keys_values[0]._k_cache._cache.size()
            # k_size = (x.shape[0], x.shape[1], x.shape[1], 1)
            # v_size = past_keys_values[0]._v_cache._cache.size()
            v_size = (k_size[0], k_size[1], x.shape[1], k_size[3])
            past_keys_values[0].update(torch.rand(v_size), torch.rand(v_size))
        return x

        


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
import peft

def load_pretrained_model(config, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map={"": device},
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    peft_config = peft.LoraConfig(
        peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.rank,
        lora_alpha=8,
        lora_dropout=config.dropout,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    )
    base_model_peft = peft.get_peft_model(base_model, peft_config)
    base_model_peft.add_adapter("dynamics", peft_config)
    base_model_peft.set_adapter("dynamics")
    print(base_model_peft.print_trainable_parameters())
    disable_causal_mask_always()
    return base_model_peft

@contextmanager
def set_adapter(model, adapter_name):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    finally:
        model.set_adapter(old_adapter_name)

def disable_causal_mask_always():
    import transformers.models.llama.modeling_llama as modeling

    decoder_fn = modeling._make_causal_mask

    def encoder_fn(*args, **kwargs):
        return torch.zeros_like(decoder_fn(*args, **kwargs))

    modeling._make_causal_mask = encoder_fn

@contextmanager
def disable_causal_mask():
    import transformers.models.llama.modeling_llama as modeling

    decoder_fn = modeling._make_causal_mask

    def encoder_fn(*args, **kwargs):
        return torch.zeros_like(decoder_fn(*args, **kwargs))

    try:
        modeling._make_causal_mask = encoder_fn
        yield
    finally:
        modeling._make_causal_mask = decoder_fn
