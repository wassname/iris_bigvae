"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from typing import Tuple

import gym
import numpy as np
import jax
from PIL import Image
import crafter
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.play_craftax import CraftaxRenderer


def make_env(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False):
    if id.startswith('Craftax'):
        return make_craftax(id)
    if id.startswith('Crafter'):
        return make_crafter(id, size=size, max_episode_steps=max_episode_steps, done_on_life_loss=done_on_life_loss)
    if id.startswith('MiniHack'):
        return make_minihack(size=size, max_episode_steps=max_episode_steps, done_on_life_loss=done_on_life_loss)
    else:
        return make_atari(id, size, max_episode_steps, noop_max, frame_skip, done_on_life_loss, clip_reward)
    
class Gymax2GymWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        gym.Wrapper.__init__(self, env)
        self.env_params = env.default_params
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        self.rngs = jax.random.split(_rng, 3)
        self.do_render = True
        if self.do_render:
            self.renderer = CraftaxRenderer(self.env, self.env_params, pixel_render_size=1)

    def step(self, action):
        obs, state, reward, done, info = self.env.step(self.rngs[2], self.env_state, action, self.env_params)
        self.env_state = state
        if self.do_render:
            self.renderer.update()
        return obs, reward, done, info

    @property
    def action_space(self):
        return self.env.action_space(self.env_params)
    
    @property
    def observation_space(self):
        return self.env.observation_space(self.env_params)
    
    def reset(self):
        obs, state = self.env.reset(self.rngs[0], self.env_params)
        self.env_state = state
        if self.do_render:
            self.renderer.update()
        return obs
    
    def render(self):
        self.renderer.render(self.env_state)


def make_craftax(id, size=64, max_episode_steps=None, done_on_life_loss=False):
    env =  make_craftax_env_from_name(id)
    env = Gymax2GymWrapper(env) # # (130, 110, 3) or (9, 11, 83)
    # if 'pixel' in id:
        # env = ResizeObsWrapper(env, (size, size))
    return env

def make_atari(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False):
    env = gym.make(id)
    print(env.spec)
    assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in str(env.spec)
    env = ResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env

def make_crafter(id, size=64, max_episode_steps=None, done_on_life_loss=False):
    # https://github.com/danijar/dreamerv2/blob/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/common/envs.py#L242
    # https://github.com/footoredo/torchbeast/blob/12939569cc46b6a8616e4c25b138d97248cc8581/torchbeast/atari_wrappers.py#L301
    env = gym.make(id) 
    env = ResizeObsWrapper(env, (size, size))
    return env


def make_minihack(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False):
    # https://github.com/facebookresearch/minihack/blob/47065748f04714c49ba5b52fb74d166228c7acc1/minihack/agent/common/envs/wrapper.py#L117
    # https://github.com/roger-creus/SOFE/blob/5551a115a9c7e1d632cf6996bf5dcabde59cdcc5/e3b/minihack/torchbeast/src/utils.py#L110
    env = gym.make(id,
                   # https://minihack.readthedocs.io/en/latest/getting-started/observation_spaces.html
                   observation_keys=("pixel_crop"),
                #    obs_crop_h=9,
                #    obs_crop_w=9,
                    )
    env = ResizeObsWrapper(env, (size, size))
    return env


class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
