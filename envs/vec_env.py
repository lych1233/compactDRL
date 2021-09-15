import cloudpickle
from multiprocessing import Process, Pipe

import numpy as np

from .base import BaseEnv


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

class SubprocVecEnv(BaseEnv):
    """The vectorized environment should inherit all necessary methods and properties of the normal base environment
    We activate a subprocess for each single environment and communicate with them as remote workers, and we always run them asynchronously but the environments should be aligned by the time step
    """
    def __init__(self, env_fns):
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
    
    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])
    
    def step_async(self, actions):
        self.waiting = True
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, reward, done, info = zip(*results)
        return np.stack(obs), np.stack(reward), np.stack(done), {"vec_info": info}
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def train(self):
        for remote in self.remotes:
            remote.send(("train", None))
    
    def eval(self):
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

class DummyVecEnv(BaseEnv):
    """The vectorized environment should inherit all necessary methods and properties of the normal base environment
    We run the environment in a serialized way in one process
    """
    def __init__(self, env_fns):
        self.num_env = len(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]
        self.continuous_action_space = self.envs[0].continuous_action_space
        self.n_obs, self.n_act = self.envs[0].n_obs, self.envs[0].n_act
    
    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions):
        results = [env.step(actions[k]) for k, env in enumerate(self.envs)]
        old_obs, reward, done, info = zip(*results)
        new_obs = [env.reset() if done else obs for obs, done, env in zip(old_obs, done, self.envs)]
        return np.stack(new_obs), np.stack(reward), np.stack(done), {"vec_info": info}
    
    def train(self):
        for env in self.envs:
            env.train()
    
    def eval(self):
        for env in self.envs:
            env.eval()
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        for env in self.envs:
            env.close()
