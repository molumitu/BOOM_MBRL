import numpy as np
import gymnasium as gym
from boom.envs.wrappers.time_limit import TimeLimit

import metaworld


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        obs = obs.astype(np.float32)
        self.env.step(np.zeros(self.env.action_space.shape))
        return obs

    def step(self, action):
        reward = 0
        for _ in range(2):
            obs, r, _, info = self.env.step(action.copy())
            reward += r
        obs = obs.astype(np.float32)
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render(
            offscreen=True, resolution=(384, 384), camera_name=self.camera_name
        ).copy()


def make_env(cfg):
    """
    Make Meta-World environment.
    """
    # Convert task name from mw-button-press to button-press-v3
    task_name = cfg.task.split("-", 1)[-1] if "-" in cfg.task else cfg.task
    env_id = f"{task_name}-v3"

    if not cfg.task.startswith("mw-"):
        raise ValueError("MetaWorld tasks must start with 'mw-'")

    # Check if environment exists in ML1
    if env_id not in metaworld.ML1.ENV_NAMES:
        raise ValueError(f"Unknown task: {cfg.task}. Available tasks: {metaworld.ML1.ENV_NAMES}")

    assert cfg.obs == "state", "This task only supports state observations."

    # Create ML1 benchmark and environment
    benchmark = metaworld.ML1(env_id)
    env = benchmark.train_classes[env_id]()
    env = MetaWorldWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
