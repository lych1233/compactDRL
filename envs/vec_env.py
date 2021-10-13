import argparse
import cloudpickle
from multiprocessing import Process, Pipe

import numpy as np

from .base import BaseEnv


class RunningNormalizer(object):
    def __init__(self, shape):
        self.shape = shape
        self.running_mean = np.zeros(shape)
        self.running_var, self.running_std = np.ones(shape), np.ones(shape)
        self.count = 1e-8
    
    def update(self, obs):
        if len(obs.shape) == len(self.shape):
            obs = np.expand_dims(obs, 0)
        batch_mean, batch_var = obs.mean(0), obs.var(0)
        batch_size = obs.shape[0]
        self._update_batch(batch_mean, batch_var, batch_size)
    
    def _update_batch(self, batch_mean, batch_var, batch_size):
        new_count = self.count + batch_size
        delta_mean = batch_mean - self.running_mean
        self.running_mean += delta_mean * batch_size / new_count
        square_sum = self.running_var * self.count + batch_var * batch_size + (delta_mean ** 2) * self.count * batch_size / new_count
        self.running_var = square_sum / new_count
        self.running_std = np.sqrt(self.running_var + 1e-8)
        self.count = new_count
    
    def normalize(self, obs):
        return (obs - self.running_mean) / self.running_std
    
    def denormalize(self, obs):
        return self.running_mean + obs * self.running_std

class CloudpickleWrapper(object):
    """Couldpickle allows more general object serialization
    """
    def __init__(self, a):
        self.a = a
    
    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, a):
        self.a = cloudpickle.loads(a)

def _worker(remote, parent_remote, env_fn_wrapper):
    """Subprocessed workers of the environment
    """
    parent_remote.close()
    env = env_fn_wrapper.a()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "train":
                env.train()
            elif cmd == "eval":
                env.eval()
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_base_info":
                remote.send((env.continuous_action_space, env.n_obs, env.n_act))
            else:
                raise NotImplementedError
        except EOFError:
            break

class VecEnv(BaseEnv):
    """The vectorized environment contains the  observation/reward normalization part
    """
    def __init__(self):
        super(VecEnv, self).__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenario", default="control", choices=["control", "atari", "mujoco"])
        parser.add_argument("--gamma", default=0.99, type=float)
        args = parser.parse_known_args()[0]
        self.gamma = args.gamma
        self.normalize_obs = args.scenario in ["control", "mujoco"]
        self.normalize_reward = args.scenario in ["control", "atari", "mujoco"]
    
    def init_normalizer(self):
        self.obs_normalizer = RunningNormalizer(self.n_obs) if self.normalize_obs else None
        self.ret = np.zeros(self.num_env)
        self.ret_normalizer = RunningNormalizer((self.num_env, )) if self.normalize_reward else None
    
    def process_obs(self, obs):
        if self.obs_normalizer:
            self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize(obs)
        return obs
        
    def process_reward(self, reward, done):
        self.ret = reward + self.gamma * self.ret
        if self.normalize_reward:
            self.ret_normalizer.update(self.ret)
            reward /= self.ret_normalizer.running_std
        self.ret[done] = 0
        return reward
    
    @property
    def reward_scale_factor(self):        
        return self.ret_normalizer.running_std if self.normalize_reward else 1

class DummyVecEnv(VecEnv):
    """The vectorized environment should inherit all necessary methods and properties of the normal base environment
    For DummyVecEnv, the environments are actually parallelized virtually in a serialized way in one process
    """
    def __init__(self, env_fns):
        super(DummyVecEnv, self).__init__()
        self.num_env = len(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]
        self.continuous_action_space = self.envs[0].continuous_action_space
        self.n_obs, self.n_act = self.envs[0].n_obs, self.envs[0].n_act
        self.init_normalizer()
    
    def reset(self):
        obs = np.stack([env.reset() for env in self.envs])
        self.ret = np.zeros(self.num_env)
        return self.process_obs(obs)

    def step(self, actions):
        results = [env.step(actions[k]) for k, env in enumerate(self.envs)]
        old_obs, reward, done, info = zip(*results)
        new_obs = [env.reset() if done else obs for obs, done, env in zip(old_obs, done, self.envs)]
        obs, reward, done = np.stack(new_obs), np.stack(reward), np.stack(done)
        obs, reward = self.process_obs(obs), self.process_reward(reward, done)
        return obs, reward, done, {"vec_info": info}
    
    def train(self):
        self.training = True
        for env in self.envs:
            env.train()
    
    def eval(self):
        self.training = False
        for env in self.envs:
            env.eval()
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        for env in self.envs:
            env.close()

class SubprocVecEnv(VecEnv):
    """The vectorized environment should inherit all necessary methods and properties of the normal base environment
    We activate a subprocess for each single environment and communicate with them as remote workers, and we always run them asynchronously but the environments should be aligned by time step
    """
    def __init__(self, env_fns):
        super(SubprocVecEnv, self).__init__()
        self.waiting = False
        self.closed = False
        self.num_env = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_env)])
        self.ps = [
            Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)), daemon=True)
            for remote, work_remote, env_fn in zip(self.remotes, self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        for remote in self.work_remotes:
            remote.close()
        
        self.remotes[0].send(("get_base_info", None))
        self.continuous_action_space, self.n_obs, self.n_act = self.remotes[0].recv()
        self.init_normalizer()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = np.stack([remote.recv() for remote in self.remotes])
        self.ret = np.zeros(self.num_env)
        return self.process_obs(obs)
    
    def step_async(self, actions):
        self.waiting = True
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, reward, done, info = zip(*results)
        obs, reward, done = np.stack(obs), np.stack(reward), np.stack(done)
        obs, reward = self.process_obs(obs), self.process_reward(reward, done)
        return obs, reward, done, {"vec_info": info}
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def train(self):
        self.training = True
        for remote in self.remotes:
            remote.send(("train", None))
    
    def eval(self):
        self.training = False
        for remote in self.remotes:
            remote.send(("eval", None))
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        if self.closed: return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True
