# Implements a Replay Buffer for experience replay.
# This is a key component for off-policy and online learning algorithms.
import random
from collections import namedtuple, deque

# The 'obs' is the raw observation from the environment before cortex processing.
# 'h' and 'z' are the world model states *before* the action was taken.
# 'activation_path' is the STAG activation path corresponding to the state (h, z).
Experience = namedtuple('Experience',
                        ('h', 'z', 'activation_path', 'obs', 'action', 'log_prob', 'reward', 'next_obs', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
