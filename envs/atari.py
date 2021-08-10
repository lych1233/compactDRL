import argparse
from collections import deque

import atari_py
import cv2
import numpy as np
import torch
from torch.functional import Tensor

from envs import BaseEnv


class AtariALE(BaseEnv):
    """Atati environment based on Atari ALE, wrapped in a OpenAI Gym API manner
    The observation will be a float tensor of S * L * L in range [0, 1) 
    
    Atari configutation
    ---------------------------------------------------------------------------
    screen size "L"      |   clip the image into L*L squares
    sliding window "S"   |   keep a series of contiguous frames as one observation (represented as a S*L*L tensor)
    max episode length   |   the maximum steps in one episode to enforce an early stop in some case
    """  
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="atari parser")
        parser.add_argument("--screen-size", type=int, default=84, \
            help="clip the image into L*L squares")
        parser.add_argument("--sliding-window", type=int, default=4, \
            help="keep a series of contiguous frames as one observation (represented as a S*L*L tensor)")
        parser.add_argument("--max-episode-length", type=int, default=10000, \
            help="the maximum steps in one episode to enforce an early stop in some case")
        return parser.parse_known_args()[0]

    def __init__(self, env):
        super(AtariALE, self).__init__()
        args = self.get_args()

        self.ale = atari_py.ALEInterface()
        self.ale.setFloat("repeat_action_probability", 0)
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        self.ale.setInt("random_seed", np.random.randint(1e9))
        self.ale.setInt("max_num_frames_per_episode", args.max_episode_length)
        self.ale.loadROM(atari_py.get_game_path(env))  # This should be after setting all parameters

        action_space = self.ale.getMinimalActionSet()
        self.real_actions = dict([i, a] for i, a in enumerate(action_space))
        self.side_len = args.screen_size
        self.lives = 0
        self.fake_death = False
        self.window = args.sliding_window
        self.n_obs, self.n_act = (self.window, self.side_len, self.side_len), len(action_space)
        self.observation = deque([], maxlen=self.window)
        self.training = True # In training mode, an episode terminates whenever the agent dies

    def get_one_frame(self):
        frame = cv2.resize(self.ale.getScreenGrayscale(), (self.side_len, self.side_len), interpolation=cv2.INTER_LINEAR)
        return np.array(frame, dtype=np.float32) / 256
    
    def reset(self):
        if self.fake_death:
            self.fake_death = False
            self.ale.act(0)
        else:
            for _ in range(self.window):
                self.observation.append(np.zeros((self.side_len, self.side_len), dtype=np.float32))
            self.ale.reset_game()
            for _ in range(np.random.randint(30)):
                self.ale.act(0)
                if self.ale.game_over(): self.ale.reset_game()
        self.observation.append(self.get_one_frame())
        self.lives = self.ale.lives()
        return np.stack(list(self.observation), 0)

    def step(self, action):
        action = self.get_proper_action(action)
        frames = np.zeros((2, self.side_len, self.side_len), dtype=np.float32)
        reward, done = 0, False
        for _ in range(4):
            reward += self.ale.act(self.real_actions[action])
            if _ >= 2:
                frames[_ - 2] = self.get_one_frame()
            if self.ale.game_over():
                done = True
                break
        self.observation.append(frames.max(0))
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0: # In Q-bert, there will be some additional frames when all lives are lost
                self.fake_death = not done
                self.lives = lives
                done = True
        return np.stack(list(self.observation), 0), reward, done, {"lives": self.lives}

    def render(self):
        cv2.imshow("screen", self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
    
    @staticmethod
    def get_proper_action(action):
        if isinstance(action, (np.ndarray, torch.Tensor)):
            action = action.item()
        if isinstance(action, (list, tuple)):
            action = np.array(action).item()
        if isinstance(action, int):
            return action
        else:
            raise TypeError("Only single value with type [int, np.ndarray, torch.Tensor, list, tuple] is allowed")
    