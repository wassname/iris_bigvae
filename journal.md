# 2023-11-03 11:36:39

Step 1 trying to get a VAE working

ah bitsandbytes
so I needed to use a `poetry add https://github.com/TimDettmers/bitsandbytes/releases/download/0.41.0/bitsandbytes-0.41.0-py3-none-any.whl` to get it to work

# 2023-11-09 07:29:27

Where am I up to?

- [ ] I want to get bigVAE working with Mistral
  - [ ] run
  - [ ] Then simplify it
- [ ] then understand IRIS, and inser this model


Eror: OOM w minstral. It runs out of mem when set_adapter, with an input
- 7.6/24 at first stop point we have
  - vae: DecoderOnlyTransformerVAE
  - self.model
  - and we seem to be running inputs through with grad
  - which is werid as it's jsut
    - a frozen model
    - 2 adaptors (float32)
    - and a vae head
  - `next(iter(self.model.parameters()))` is bfloat16, cuda:0. As is vae

    $ self.model
    PeftModel(
    (base_model): LoraModel(
        (model): MistralForCausalLM(
        (model): MistralModel(
            (embed_tokens): Embedding(32000, 4096)

    $ self.vae
    DecoderOnlyTransformerVAE(
    (model): PeftModel(
        (base_model): LoraModel(
        (model): MistralForCausalLM(
            (model): MistralModel(
            (embed_tokens): Embedding(32000, 4096)
    (vae): VAEComponent(
        (f): Linear(in_features=4096, out_features=1, bias=True)
        (w_e): Linear(in_features=4096, out_features=768, bias=True)
        (w_d): Linear(in_features=768, out_features=4096, bias=True)
    )
    )



Hm JDP said he uses 8xH100 so 64GB*8. 
- A p5.48xlarge is 80$
- a o2.16x is 192GB and $14/h
we are using peft

```py
self.model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 7,577,276,416 || trainable%: 1.107074302091819


self.model
```

    DecoderOnlyTransformerVAE(
    (model): PeftModel(
        (base_model): LoraModel(
        (model): MistralForCausalLM(
            (model): MistralModel(
            (embed_tokens): Embedding(32000, 4096)
            (layers): ModuleList(
                (0-31): 32 x MistralDecoderLayer(
                (self_attn): MistralAttention(           )
                (mlp): MistralMLP()
                (input_layernorm): MistralRMSNorm()
                (post_attention_layernorm): MistralRMSNorm()
                )
            )
            (norm): MistralRMSNorm()
            )
            (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
        )
        )
    )
    (vae): VAEComponent(
        (f): Linear(in_features=4096, out_features=1, bias=True)
        (w_e): Linear(in_features=4096, out_features=768, bias=True)
        (w_d): Linear(in_features=768, out_features=4096, bias=True)
    )
    )

How much gpu ram should it take it train a 7B?
- 3B -> 14
- 12B - 56GB
- so 7B should be ~30 :(. or 20 with cpu offloading
- batch size of 4... with 1 it takes 20GB then crashes

# 2023-11-10 11:41:21

tldr:
- I can't use Mistral without a bigger gpu: 30GB+, or maybe I can use the deepspeed gpu offloading (batch=1)
- I I can just use GPT2 like adavae. Or stablelm

Ideally I can use a small one for prototyping, and change to a large one if it works.


Models:
- mistral: the BigVAE code is setup for it
- gpt2: the AdaVAE code is setup for it... but it's also way messier. Roll your own adapter etc


TODO use no_grad! way better


now
- look at training code
- what is JP code actually doing? start with generate topic
- clean it up, there is so much repeeated code


# 2023-11-11 10:09:50

## Now how does VAE generate work inside?

The VAE generate function takes in context, input embeddings, target embeddings, and other parameters in that order. The function performs the following steps:
- Encode: The input embeddings are encoded using an adapter on `self.model` and `vaecomponent.encoder`. This step involves using linear and pooling operations, followed by a softmax function to sample and obtain `z`.
- Decode/Embed: `z` is decoded to obtain `z_embed`, and the decoder is provided with target ID embeddings. This step is done using a linear operation and the model embeddings.
- Model: The model is run on the embeddings to obtain logits. The embeddings can be considered as latent states, and the latent state is expressed in the language of embeddings.

# 2023-11-12 08:06:25

Now look at IRIS and TWM world models.

- https://github.com/eloialonso/iris/blob/main/src/models/world_model.py
  - Model(x, kv_cache). Where x is B, T, C. Batch, Time, Channels?
  - if just takes in x and outputs x. The output are logits, from a linear layer.
- https://github.com/jrobine/twm/blob/main/twm/world_model.py



# 2023-11-12 08:46:46

Adding BigVAe as transformer layer

I can't tokenize, then pass in input id's. As I need to to be backpropable. So I need to by pass the embedding layers...
- perhaps I can encode actions by things I have previously embedded?

# 2023-11-12 10:10:01

What the diff between BigVAERouter and DecoderOnlyTransformerVAE

- DecoderOnlyTransformerVAE(prefix_ids, input_ids) -> outputs, mean
- BigVAERouter(prefix_ids, input_ids, target_ids,), 

OK I want to rename here:
- prefix_ids 
- embed_ids -> input_ds
- target_ids
- decoder_prefix_ids -> prefix_ids

Now how to reconcille it with the world model. what does the world model do?
- components
  - transforer x->x: the part we are replacing
  - embedder: a custom embedder
  - heads for each output: obs, reward, end_of_episode
- when it goes foward
  -  tokens -> (x, obs, rewards, ends)
     -  where is is the output of the transformer/vae
  -  x = transformer(sequences)
  -  where the sequences are tokens embedded with the embedder. 
     -  TODO: I will want to use the model embedder if possible?

## Embedder deep dive

First the code

So the embedder, takes in `tokens` then breaks it up using a slices, into a seperate obs and action embedder.
We add them, plus positions, and add them into one of embed_dim

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))


Now how does the paper describe it? https://openreview.net/pdf?id=vhFu1Acb0xb


# IDEAS

Could I just use a single transformer? Dreamer seem use encode and decode to a latent state, but IRIS doesn't?
