from collections import namedtuple

Experience = namedtuple('Experience', ['h_t', 'z_t', 'activation_path', 'obs', 'action', 'log_prob', 'reward', 'next_obs', 'done', 'goal', 'winner_id'])
