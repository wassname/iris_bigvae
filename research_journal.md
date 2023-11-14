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

# 2023-11-12 16:58:37

So I got it training, but during imagination it passes in a single token with no past steps. But the slicer seems to need at least on block? And so I get none?

hmm it's because num_kept_tokens is 16 not 1. So there should be a whole block passed in ?

wait apparently it's also a problem in the normal repo.... I confuse! maybe it's my config! maybe I need >larger than block size. nope

hmm it still happens in the original repo with my debug params. maybe it's my debug params


... trying a full run without my debug params...

note trains.world_model.batch_num_samples:4 fill 20GB gpu ram for the 3b stability ai llm

ok even with a full run I get the error. I think it's a bug in the original repo. I'll try to debug it there.

    Epoch 51 / 600

    Experience collection (train_dataset): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:03<00:00, 59.91it/s]
    Training tokenizer: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:17<00:00, 11.53it/s]
    Training world_model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [02:11<00:00,  1.53it/s]
    Training actor_critic:   0%|                                                                                                                                                                                     | 0/200 [00:00<?, ?it/s]
    Error executing job with overrides: ['env.train.id=BreakoutNoFrameskip-v4', 'common.device=cuda:0', 'wandb.mode=offline']
    Traceback (most recent call last):
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/src/main.py", line 10, in main
        trainer.run()
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/src/trainer.py", line 111, in run
        to_log += self.train_agent(epoch)
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/src/trainer.py", line 146, in train_agent
        metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False, tokenizer=self.agent.tokenizer, world_model=self.agent.world_model, **cfg_actor_critic)
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/src/trainer.py", line 161, in train_component
        losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/src/models/actor_critic.py", line 102, in compute_loss
        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/src/models/actor_critic.py", line 149, in imagine
        obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1))
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/.venv/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
        return func(*args, **kwargs)
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/src/envs/world_model_env.py", line 75, in step
        reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/.venv/lib/python3.9/site-packages/torch/distributions/categorical.py", line 70, in __init__
        super().__init__(batch_shape, validate_args=validate_args)
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/.venv/lib/python3.9/site-packages/torch/distributions/distribution.py", line 66, in __init__
        valid = constraint.check(value)
    File "/media/wassname/SGIronWolf/projects5/worldmodels/iris_bigvae/.venv/lib/python3.9/site-packages/torch/distributions/constraints.py", line 226, in check
        result = result.reshape(
    RuntimeError: cannot reshape tensor of 0 elements into shape [8, 0, -1] because the unspecified dimension size -1 can be any value and is ambiguous

Oh maybe it's because we don't keep track of KV cache, but it's actually used to track number of steps!!

# 2023-11-13 20:11:51

I go it working byt ut takes 30 seconds for one one actor critic batch, werird

Experience collection (train_dataset): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:03<00:00, 60.45it/s]
Training tokenizer: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:17<00:00, 11.26it/s]
Training world_model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [02:12<00:00,  1.51it/s]
Training actor_critic:  82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                             | 165/200 [1:01:43<13:24, 22.99s/it]


hm maybe it's just the face it has to backprop throguh the whole LLM :( damn... is there another way to train it? Daym. How many params did the original have?

well running eval on the transformer brought it down from 100sec to 60, but it's still huge. 

But then why is the model training fast? It makes not sense
