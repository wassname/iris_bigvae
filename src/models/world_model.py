from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import Batch
from src.models.kv_caching import KeysValues
from src.models.slicer import Embedder, Head
from src.models.tokenizer import Tokenizer
from src.models.transformer import Transformer, TransformerConfig
from src.utils import init_weights, LossWithIntermediateLosses
from src.models.bigvae import BigVAEConfig


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: BigVAEConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config       

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern
        
        self.transformer = Transformer(config)
        transformer_embedding = self.transformer.embedding

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        self.act_emb = nn.Embedding(act_vocab_size, config.embed_dim)

        # FIXME: having slices is unclear. maybe it's better just to have obs and action embeddings?
        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([self.act_emb, transformer_embedding])
        )
        
        # why have this? Well I worry that the transformer can't adapt, since so much is frozen
        # TODO: If I get the dynamics model working, maybe try without it
        self.post_embed = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            # nn.ReLU(),
            # nn.Linear(config.embed_dim, config.embed_dim)
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        # don't apply to transformer or transformer/obs embeddings
        self.act_emb.apply(init_weights)
        self.pos_emb.apply(init_weights)
        self.post_embed.apply(init_weights)
        self.head_observations.apply(init_weights)
        self.head_rewards.apply(init_weights)
        self.head_ends.apply(init_weights)
        
        

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        num_steps = tokens.size(1)  # (B=8, T=170) where often the last 10 are actons
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))
        # [batch=8, num_steps=170, embed_size=2048]
        sequences = self.post_embed(sequences)
        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        # with torch.no_grad():
        # [B=8, S=10, Colors=3, H=64, W=64] -> [B=8, S=10, 16]
        obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)

        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
        # So first 10 are observation, the last 10 tokens are actions

        outputs = self(tokens)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
