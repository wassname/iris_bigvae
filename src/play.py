from functools import partial
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from src.agent import Agent
from src.envs import SingleProcessEnv, WorldModelEnv
from src.game import AgentEnv, EpisodeReplayEnv, Game
from src.models.actor_critic import ActorCritic
from src.models.world_model import WorldModel
from src.models.tokenizer import Tokenizer


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    device = torch.device(cfg.common.device)
    assert cfg.mode in (
        "episode_replay",
        "agent_in_env",
        "agent_in_world_model",
        "play_in_world_model",
    )

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    if cfg.mode.startswith("agent_in_"):
        h, w, _ = test_env.env.unwrapped.observation_space.shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    if cfg.mode == "episode_replay":
        env = EpisodeReplayEnv(
            replay_keymap_name=cfg.env.keymap, episode_dir=Path("media/episodes")
        )
        keymap = "episode_replay"

    else:
        # tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(
            obs_vocab_size=cfg.tokenizer.vocab_size,
            act_vocab_size=test_env.num_actions,
            config=instantiate(cfg.world_model),
        )
        transformer_embedding = world_model.transformer.embedding
        tokenizer = Tokenizer(
            transformer_embedding=transformer_embedding,
            vocab_size=cfg.tokenizer.vocab_size,
            embed_dim=cfg.tokenizer.embed_dim,
            encoder=instantiate(cfg.tokenizer.encoder),
            decoder=instantiate(cfg.tokenizer.decoder),
        )
        actor_critic = ActorCritic(
            **cfg.actor_critic, act_vocab_size=test_env.num_actions
        )
        agent = Agent(tokenizer, world_model, actor_critic).to(device)
        agent.load(Path("checkpoints/last.pt"), device)

        if cfg.mode == "play_in_world_model":
            env = WorldModelEnv(
                tokenizer=agent.tokenizer,
                world_model=agent.world_model,
                device=device,
                env=env_fn(),
            )
            keymap = cfg.env.keymap

        elif cfg.mode == "agent_in_env":
            env = AgentEnv(
                agent, test_env, cfg.env.keymap, do_reconstruction=cfg.reconstruction
            )
            keymap = "empty"
            if cfg.reconstruction:
                size[1] *= 3

        elif cfg.mode == "agent_in_world_model":
            wm_env = WorldModelEnv(
                tokenizer=agent.tokenizer,
                world_model=agent.world_model,
                device=device,
                env=env_fn(),
            )
            env = AgentEnv(agent, wm_env, cfg.env.keymap, do_reconstruction=False)
            keymap = "empty"

    game = Game(
        env,
        keymap_name=keymap,
        size=size,
        fps=cfg.fps,
        verbose=bool(cfg.header),
        record_mode=bool(cfg.save_mode),
    )
    game.run()


if __name__ == "__main__":
    main()
