"""
Fine-tunes a language model on pre-tokenized data.


From https://raw.githubusercontent.com/JD-P/minihf/adavae-moe/vae_infer.py
See https://huggingface.co/jdpressman/BigVAE-Mistral-7B-v0.1/blob/main/README.md

BigVAE is an [AdaVAE](https://arxiv.org/abs/2205.05862) trained as a pair of LoRa finetunes on [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1).
It is meant to be used with the [MiniHF VAE inference code](https://github.com/JD-P/minihf/blob/adavae-moe/vae_infer.py) and will not work if you try to load it....
"""
from dataclasses import dataclass
import argparse
from contextlib import contextmanager
from itertools import chain, islice
import json
import math
from pathlib import Path
import random
import os
import sys
import zipfile

import accelerate
# from datasets import load_dataset
import peft
import safetensors.torch as safetorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger
from peft import PeftModel, LoraConfig

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
print = tqdm.external_write_mode()(logger.info)


def cosine_warmup(steps, value=1.0):
    return lambda i: value * math.sin(min(i / steps, 1) * math.pi / 2) ** 2


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


def gumbel_like(x):
    return torch.rand_like(x).log_().nan_to_num_().neg_().log_().neg_()


@contextmanager
def disable_causal_mask():
    raise NotImplementedError("FIXME")
    import transformers.models.llama.modeling_llama as modeling

    decoder_fn = modeling._make_causal_mask

    def encoder_fn(*args, **kwargs):
        return torch.zeros_like(decoder_fn(*args, **kwargs))

    try:
        modeling._make_causal_mask = encoder_fn
        yield
    finally:
        modeling._make_causal_mask = decoder_fn


@contextmanager
def disable_causal_mask_mistral():
    raise NotImplementedError("FIXME")
    import transformers.models.mistral.modeling_mistral as modeling

    decoder_fn = modeling._make_sliding_window_causal_mask

    def encoder_fn(*args, **kwargs):
        return torch.zeros_like(decoder_fn(*args, **kwargs))

    try:
        modeling._make_sliding_window_causal_mask = encoder_fn
        yield
    finally:
        modeling._make_sliding_window_causal_mask = decoder_fn
        

class VAEHead(nn.Module):
    def __init__(self, d_model, z_dim):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.f = nn.Linear(d_model, 1)
        self.w_e = nn.Linear(d_model, z_dim)
        self.w_d = nn.Linear(z_dim, d_model)
        nn.init.orthogonal_(self.w_e.weight)
        with torch.no_grad():
            self.w_d.weight.copy_(self.w_e.weight.T)

    def encode(self, hidden_states, attention_mask):
        scores = self.f(hidden_states)
        scores = scores + attention_mask[:, :, None].log().nan_to_num()
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights, dim=1)
        return self.w_e(pooled)

    def sample(self, mean, tau=1.0):
        return mean + torch.randn_like(mean) * tau**0.5

    def decode(self, z):
        return self.w_d(z)


class BigVAE(nn.Module):
    """
    Version of AdaVAE with transformer. 
    """
    
    def __init__(self, base_model_peft: PeftModel, peft_config: LoraConfig, z_dim: int=768, device: str = "cuda"):
        super().__init__()
        self.model = base_model_peft
        self.model.add_adapter("encoder", peft_config)
        self.model.add_adapter("decoder", peft_config)
        self.model.config.output_hidden_states = True
        self.vae_head = VAEHead(self.model.config.hidden_size, z_dim).to(device)

    def save_pretrained(self, path):
        path = Path(path)
        self.model.save_pretrained(path, safe_serialization=True)
        safetorch.save_file(self.vae_head.state_dict(), path / "vae.safetensors")

    def load_pretrained(self, path, is_trainable=False):
        path = Path(path)
        self.model.delete_adapter("encoder")
        self.model.load_adapter(path / "encoder", "encoder", is_trainable=is_trainable)
        self.model.delete_adapter("decoder")
        self.model.load_adapter(path / "decoder", "decoder", is_trainable=is_trainable)
        self.vae_head.load_state_dict(safetorch.load_file(path / "vae.safetensors"))

    def encode(self, input_ids, attention_mask):
        with set_adapter(self.model, "encoder"), disable_causal_mask_mistral():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        return self.vae_head.encode(outputs.hidden_states[-1], attention_mask)

    def input_ids_to_embeds(self, input_ids):
        embed_weight = self.model.get_input_embeddings().weight
        input_one_hots = F.one_hot(input_ids, num_classes=self.model.config.vocab_size)
        return input_one_hots.to(embed_weight) @ embed_weight

    @torch.no_grad()
    def generate(self, z, input_ids, attention_mask, n_tokens, tau=1.0):
        """
        Takes in a latent vector z from past tokens and generates next n_tokens tokens.
        
        Used in e.g. https://github.com/JD-P/minihf/blob/adavae-moe/train_vae_overlap.py#L335
        """
        z_embed = self.vae_head.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids)
        inputs_embeds = torch.cat([z_embed, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask.new_ones([attention_mask.shape[0], 1]), attention_mask], dim=1
        )
        new_embeds, past = None, None
        with set_adapter(self.model, "decoder"):
            for _ in range(n_tokens):
                outputs = self.model(
                    inputs_embeds=inputs_embeds if past is None else new_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
                logits = outputs.logits[:, -1:, :].float()
                new_input_ids = torch.argmax(logits + gumbel_like(logits) * tau, dim=-1)
                input_ids = torch.cat([input_ids, new_input_ids], dim=1)
                new_embeds = self.input_ids_to_embeds(new_input_ids)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
                )
                past = outputs.past_key_values
        return input_ids

    def forward(self, input_ids, attention_mask, prefix_ids, prefix_mask):
        input_ids_all = torch.cat([prefix_ids, input_ids], dim=1)
        attn_mask_all = torch.cat([prefix_mask, attention_mask], dim=1)
        mean = self.encode(input_ids, attention_mask)
        z = self.vae_head.sample(mean)
        z_embed = self.vae_head.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids_all)
        inputs_embeds = torch.cat([z_embed, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask.new_ones([attn_mask_all.shape[0], 1]), attn_mask_all], dim=1
        )
        with set_adapter(self.model, "decoder"):
            outputs = self.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
            )
        return outputs, mean

class BigVAERouter(nn.Module):
    def __init__(self, base_model_peft: PeftModel, vae: BigVAE, device: str = "cuda"):
        super().__init__()
        peft_config = base_model_peft.peft_config["default"] # debug this
        self.model = base_model_peft
        self.model.add_adapter("router", peft_config)
        self.model.config.output_hidden_states = True
        self.vae = vae

    def save_pretrained(self, path):
        path = Path(path)
        self.model.save_pretrained(path, safe_serialization=True)
        safetorch.save_file(self.model.state_dict(), path / "router.safetensors")
        safetorch.save_file(self.vae.vae_head.state_dict(), path / "vae.safetensors")

    def load_pretrained(self, path, is_trainable=False):
        path = Path(path)
        self.model.delete_adapter("router")
        if (path / "router").exists():
            self.model.load_adapter(path / "router", "router", is_trainable=is_trainable)
        else:
            self.model.load_adapter(path / "decoder", "router", is_trainable=is_trainable)
        
    def encode(self, input_ids, attention_mask):
        with set_adapter(self.vae.model, "encoder"), disable_causal_mask_mistral():
            outputs = self.vae.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        return self.vae.vae_head.encode(outputs.hidden_states[-1], attention_mask)

    def input_ids_to_embeds(self, input_ids):
        embed_weight = self.model.get_input_embeddings().weight
        input_one_hots = F.one_hot(input_ids, num_classes=self.model.config.vocab_size)
        return input_one_hots.to(embed_weight) @ embed_weight

    def generate(self, z, input_ids, attention_mask, n_tokens, tau=1.0):
        """
        predict next token given a latent vector z and previous tokens as input_ids
        
        e.g. https://github.com/JD-P/minihf/blob/adavae-moe/vae_infer.py#L428
        """
        z_embed = self.vae.vae_head.decode(z)[:, None]
        inputs_embeds = self.input_ids_to_embeds(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, z_embed], dim=1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
        )
        new_embeds, past = None, None
        with set_adapter(self.vae.model, "router"):
            for _ in range(n_tokens):
                outputs = self.model(
                    inputs_embeds=inputs_embeds if past is None else new_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
                logits = outputs.logits[:, -1:, :].float()
                new_input_ids = torch.argmax(logits + gumbel_like(logits) * tau, dim=-1)
                input_ids = torch.cat([input_ids, new_input_ids], dim=1)
                new_embeds = self.input_ids_to_embeds(new_input_ids)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
                )
                past = outputs.past_key_values
        return input_ids

    # def generate_cfg(self, z, input_ids, attention_mask, n_tokens, tau=1.0, cfg_scale=1):
    #     """
    #     predict next tokens given a latent vector z and previous tokens as input_ids
        
    #     but this one mixes base and router
    #     was used in topic modelling here https://github.com/JD-P/minihf/blob/adavae-moe/vae_infer.py#L614
    #     I can soon delete it
    #     """
    #     z_embed = self.vae.vae.decode(z)[:, None]
    #     inputs_embeds_base = self.input_ids_to_embeds(input_ids)
    #     inputs_embeds_router = torch.cat([inputs_embeds_base, z_embed], dim=1)
    #     attention_mask = torch.cat(
    #         [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
    #     )
    #     new_embeds, base_past, router_past = None, None, None
    #     for _ in range(n_tokens):
    #         with set_adapter(self.vae.model, "router"):
    #             router_outputs = self.model(
    #                 inputs_embeds=inputs_embeds_router if router_past is None else new_embeds,
    #                 attention_mask=attention_mask,
    #                 use_cache=True,
    #                 past_key_values=router_past,
    #             )
    #         with set_adapter(self.vae.model, None):
    #             base_outputs = self.model(
    #                 inputs_embeds=inputs_embeds_base if base_past is None else new_embeds,
    #                 attention_mask=attention_mask[:,:-1],
    #                 use_cache=True,
    #                 past_key_values=base_past,
    #             )
    #             base_logits = base_outputs.logits[:, -1:, :].float()
    #             router_logits = router_outputs.logits[:, -1:, :].float()
                
    #             # mix base and router prediction based on cfg_scale
    #             logits = base_logits + cfg_scale * (router_logits - base_logits)
    #             new_input_ids = torch.argmax(logits + gumbel_like(logits) * tau, dim=-1)
    #             input_ids = torch.cat([input_ids, new_input_ids], dim=1)
    #             new_embeds = self.input_ids_to_embeds(new_input_ids)
    #             attention_mask = torch.cat(
    #                 [attention_mask, attention_mask.new_ones([attention_mask.shape[0], 1])], dim=1
    #             )
    #             base_past = base_outputs.past_key_values
    #             router_past = router_outputs.past_key_values
    #     return input_ids
    
    def forward(self, input_ids, input_mask, target_ids, target_mask, prefix_ids, prefix_mask):
        """Like the decoder only, but with context/prefix."""
        mean = self.encode(input_ids, input_mask)
        z = self.vae.vae_head.sample(mean)
        z_embed = self.vae.vae_head.decode(z)[:, None]
        prefix_embeds = self.input_ids_to_embeds(prefix_ids)
        target_embeds = self.input_ids_to_embeds(target_ids)
        inputs_embeds = torch.cat([prefix_embeds, z_embed, target_embeds], dim=1)
        attention_mask = torch.cat(
            [prefix_mask,
            target_mask.new_ones([prefix_mask.shape[0], 1]),
            target_mask], dim=1
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False
        )
        return outputs

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


@dataclass
class BigVAEConfig:
    
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float
    
    model_name: str = "stabilityai/stablelm-3b-4e1t"
    dropout: float = 0
    rank: int = 32
    z_dim: int = 768
    start_from: str = None

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks

def load_model(config, device='cuda'):
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
    vae_model = BigVAE(
        base_model_peft, device, peft_config, z_dim=config.z_dim,
    )
    if config.start_from:
        vae_model.load_pretrained(config.start_from)
    base_model_peft.requires_grad_(False)
    vae_model.vae_head.requires_grad_(False)
    vae_model.vae_head.w_d.requires_grad_()
    router = BigVAERouter(base_model_peft, vae_model, device, peft_config)
    if config.start_from:
        router.load_pretrained(config.start_from, is_trainable=True)
    print(router.model.print_trainable_parameters())
    router.model.set_adapter("router")
