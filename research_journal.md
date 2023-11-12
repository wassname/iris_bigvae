# 2023-11-12 13:17:35

Try IRIs but with pretrained transformer with LoRA adapter

- [x] first can I run it yes with a 1/2 batch size
- [ ] then can I add 3B with adapter...

```sh
poetry install
. ./.venv/bin/activate
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0 wandb.mode=offline

# or for quick debug
WANDB_MODE=disabled python -m pdb src/main.py env.train.id=BreakoutNoFrameskip-v4
```


```sh
# TODO use this code to load a transformer, and other code from my bigvae repo https://github.com/wassname/bigvae_wm
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
```

Debugging:
    batch['observations'].shape
    torch.Size([16, 20, 3, 64, 64])

    obs_tokens.shape
    torch.Size([16, 20, 16])

    https://vscode.dev/github/wassname/iris_bigvae/blob/just_llms2/src/models/world_model.py#L105
    tokens
    tensor([[222, 222, 222,  ..., 409,  55,   2],
            [222, 222, 222,  ..., 409, 139,   1],
            [222, 222, 222,  ..., 168, 190,   3],
            ...,
            [222, 222, 222,  ..., 168,  55,   0],
            [222, 222, 222,  ..., 237, 190,   3],
            [222, 222, 222,  ..., 168,  55,   0]], device='cuda:0')
    tokens.shape
    torch.Size([16, 340])
    where 16 is the batch size. 340 is the step size?. actions was 16,20 int

    tokens.shape int
    torch.Size([16, 340])

    sequences.shape float32
    torch.Size([16, 340, 256])

    transfrmer
    x.shape
    torch.Size([16, 340, 256])
