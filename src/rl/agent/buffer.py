import collections
import random


class ReplayBuffer:
    def __init__(self, size=50000):
        self.size = size
        self.memory = collections.deque([], maxlen=size)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        self.memory.append(transition)

    def __len__(self):
        return len(self.memory)
