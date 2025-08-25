import random
from collections import namedtuple, deque
import numpy as np
import torch

Experience = namedtuple('Experience',
                        ('h', 'z', 'activation_path', 'obs', 'action', 'log_prob', 'reward', 'next_obs', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, sequence_length=50):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.memory = []
        self.episode_ends = []
        self.position = 0

    def push(self, *args):
        """Saves an experience."""
        e = Experience(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = e
        self.position = (self.position + 1) % self.capacity
        if e.done:
            self.episode_ends.append(self.position)

    def sample(self, batch_size):
        """Samples a batch of sequences."""
        # This is a simplified sampling method. A more robust implementation would
        # handle edge cases like very short episodes or a buffer that is not yet full.
        batch = []
        for _ in range(batch_size):
            # Find a valid starting index for a sequence
            while True:
                start_index = random.randint(0, len(self.memory) - self.sequence_length)
                # Check if the sequence crosses an episode boundary
                crosses_boundary = False
                for end_pos in self.episode_ends:
                    if start_index < end_pos < start_index + self.sequence_length:
                        crosses_boundary = True
                        break
                if not crosses_boundary:
                    break

            sequence = self.memory[start_index : start_index + self.sequence_length]
            batch.append(sequence)

        # Transpose the batch of sequences and convert to numpy arrays
        batch_dict = {}
        for key in Experience._fields:
            # The 'obs' and 'next_obs' might be sequences of images, stack them carefully
            if key in ['obs', 'next_obs']:
                 batch_dict[key] = np.stack([getattr(exp, key) for seq in batch for exp in seq])
            else:
                # Other fields can be converted to numpy arrays more directly
                data = [getattr(exp, key) for seq in batch for exp in seq]
                # Check if the data is a list of tensors before trying to stack
                if isinstance(data[0], torch.Tensor):
                    batch_dict[key] = torch.stack(data).detach().cpu().numpy()
                else:
                    batch_dict[key] = np.array(data)

        # Reshape to (batch_size, sequence_length, ...)
        for key, value in batch_dict.items():
            batch_dict[key] = value.reshape(batch_size, self.sequence_length, *value.shape[1:])

        return batch_dict


    def __len__(self):
        return len(self.memory)
