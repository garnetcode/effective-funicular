import numpy as np

class GridWorld:
    """
    A simple 2D Grid World environment for reinforcement learning.

    The state is a 2D numpy array where:
    - 0: Empty space
    - 1: Wall/Obstacle
    - 2: Agent
    - 3: Goal
    - 4: Trap
    """
    def __init__(self, size=10):
        self.size = size
        self.action_space = [0, 1, 2, 3] # 0:Up, 1:Down, 2:Left, 3:Right
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.max_steps = size * 3
        self.reset()

    def reset(self):
        """Resets the environment to a new random configuration."""
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # Place agent, goal, and traps randomly, ensuring no overlap
        positions = set()
        while len(positions) < 3: # Need 3 unique spots
            positions.add((np.random.randint(0, self.size), np.random.randint(0, self.size)))

        pos_list = list(positions)
        self.agent_pos = list(pos_list[0])
        self.goal_pos = pos_list[1]
        self.trap_pos = pos_list[2]

        self.grid[self.goal_pos] = 3
        self.grid[self.trap_pos] = 4

        self.steps_taken = 0
        return self._get_state()

    def _get_state(self):
        """Returns the current state of the grid with the agent's position."""
        state = self.grid.copy()
        state[tuple(self.agent_pos)] = 2
        return state

    def step(self, action):
        """
        Executes one time step in the environment.

        Args:
            action (int): The action to take (0-3).

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1

        # Calculate potential new position
        move = self.action_map[action]
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]

        # Check for boundary collisions
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.agent_pos # Stay in place if hitting a wall

        self.agent_pos = new_pos

        # Check for terminal states
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 10.0
            done = True
        elif tuple(self.agent_pos) == self.trap_pos:
            reward = -10.0
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -2.0 # Penalty for running out of time
            done = True
        else:
            reward = -0.1 # Small penalty for each step to encourage efficiency
            done = False

        return self._get_state(), reward, done, {}

    def render(self):
        """Prints a text representation of the grid."""
        state = self._get_state()
        char_map = {0: '.', 1: '#', 2: 'A', 3: 'G', 4: 'X'}

        for row in state:
            print(' '.join([char_map[cell] for cell in row]))
        print("-" * (self.size * 2))
