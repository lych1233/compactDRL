import argparse

class BaseBuffer(object):
    """Clarify basic functions that a buffer should have

    buffer_type       |   "dequeue" is First-In-First-Out;
                          "reservoir" randomly shuffles old one;
    buffer_capacity   |   the maximum number of trainsitions the buffer can hold
    """
    
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="buffer parser")
        parser.add_argument("--buffer_type", default="dequeue", choices=["dequeue", "reservoir"], \
            help="the way to kick out old data when the buffer is full")
        parser.add_argument("--buffer_capacity", default=4096, type=int, \
            help="the maximum number of trainsitions the buffer can hold")
        return parser.parse_known_args()[0]
    
    def __init__(self):
        self.kicking = None # How to kick old data when the buffer is full
    
    def add(self, obs, action, reward, done, next_obs):
        raise NotImplementedError
        return None # Return the index where the transition is stored
    
    def get(self, idx):
        # The idx should be in numpy ndarray shape; and the acuqired data will be returned in the same shape
        raise NotImplementedError
        return None # Return data on given indexes
    
    def clearall(self):
        raise NotImplementedError