from collections import deque

import gymnasium as gym
import numpy as np
import torch


class PixelWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with DMControl environments.
    """

    def __init__(self, cfg, env, num_frames=3, render_size=64):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_frames * 3, render_size, render_size),
            dtype=np.uint8,
        )
        self._frames = deque([], maxlen=num_frames)
        self._render_size = render_size

    def _get_obs(self):
        """Render and process image observation."""
        # DMControl uses physics.render(height, width, camera_id)
        # TimeStepToGymWrapper.render accepts (mode, width, height, camera_id)
        # Default render is 640x480 (non-square), we need square 64x64
        frame = self.env.render(
            width=self._render_size, 
            height=self._render_size,
            camera_id=0
        )
        
        # Handle case where render returns tuple (new gymnasium)
        if isinstance(frame, tuple):
            frame = frame[0]
        
        # Transpose from [H, W, C] to [C, H, W]
        if frame.ndim == 3:
            frame = frame.transpose(2, 0, 1)
        
        self._frames.append(frame)
        return torch.from_numpy(np.concatenate(self._frames))

    def reset(self):
        self.env.reset()
        for _ in range(self._frames.maxlen):
            obs = self._get_obs()
        return obs, {}

    def step(self, action):
        _, reward, done, truncated, info = self.env.step(action)
        return self._get_obs(), reward, done, truncated, info
