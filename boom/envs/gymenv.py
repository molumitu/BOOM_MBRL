import os
import sys

import numpy as np
import gymnasium as gym

class GymEnv(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.max_episode_steps = getattr(self.env, "_max_episode_steps", None)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):  
        return self.env.render()

def make_env(cfg):
    env = gym.make(cfg.task, render_mode="rgb_array")  # 设置渲染模式
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=500)  # 强制500步结束
    return GymEnv(env, cfg)

if __name__ == "__main__":
    cfg = type('Config', (object,), {})()
    cfg.task = "Humanoid-v3"
    env = make_env(cfg)
    env.reset()
    for i in range(100):
        a = env.action_space.sample()
        env.step(a)
        # env.render()
    print("done")