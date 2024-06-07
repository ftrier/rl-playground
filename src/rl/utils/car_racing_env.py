import gymnasium as gym
import numpy as np


class CarRacingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        self.env = gym.make('CarRacing-v2', continuous=False,
                            render_mode=render_mode)
        self.stacked_frames = []
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (4, 42, 48), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

    def _preprocess(self, s):
        s = s[0:84, :, :].mean(axis=2)
        s = s[::2, ::2]
        return s

    def reset(self):
        s, info = self.env.reset()
        [self.env.step(0) for _ in range(50)]
        self.stacked_frames = np.array([self._preprocess(s)] * 4)
        return self.stacked_frames

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self.stacked_frames = np.roll(self.stacked_frames, -1, axis=0)
        self.stacked_frames[-1] = self._preprocess(observation)
        return self.stacked_frames, reward, terminated, truncated, info
