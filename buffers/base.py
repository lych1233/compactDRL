import argparse


class BaseBuffer(object):
    """Clarify basic functions that a buffer should have

    buffer_type       |   "dequeue" is First-In-First-Out;
                      |   "reservoir" randomly eliminates an old peice of data;
    buffer_capacity   |   the maximum number of trainsitions a buffer can hold
    """
    
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="buffer parser")
        parser.add_argument("--buffer_type", default="dequeue", choices=["dequeue", "reservoir"], \
            help="the way to kick out old data when the buffer is full")
        parser.add_argument("--buffer_capacity", default=1000000, type=int, \
            help="the maximum number of trainsitions a buffer can hold")
        return parser.parse_known_args()[0]
    
    def __init__(self):
        self.kicking = None # How to kick old data when the buffer is full
    
    def add(self, obs, action, reward, done, next_obs):
        raise NotImplementedError
        # Return the index where the transition is stored
    
    def get(self, idx):
        '''The idx should be in numpy ndarray shape; and the acuqired data will be returned in the following forms:
            obs      |   [B, n_obs]
            action   |   [B] for discrete action space; [B, n_act] for continuous action space
            reward   |   [B]
            done     |   [B]
        and all of them are in np.float32 dtype (except the discrete action vector is in np.long)
        except that for vectorized buffer we have:
            obs      |   [B, N_Env, n_obs]
            action   |   [B, N_Env] for discrete action space; [B, N_Env, n_act] for continuous action space
            reward   |   [B, N_Env]
            done     |   [B, N_Env]
        '''
        raise NotImplementedError
        # Return data on given indexes
    
    def clearall(self):
        raise NotImplementedError