import gymnasium as gym
from gymnasium.spaces import Box
from tianshou.env import DummyVectorEnv
import numpy as np
from .config import X, Y, H, M, N, P, STEPS_PER_EPISODE, NOISE_POWER, CARRIER_FREQUENCY, PATH_LOSS_EXPONENT

class CellFreeEnv(gym.Env):

    def __init__(self):
        self._num_steps = 0
        self._terminated = False

        # Initialize base station positions (uniformly distributed)
        self.bs_positions = np.random.uniform(0, [X, Y, 0], (M, 3))  # [x, y, z] for each BS

        # Initialize UAV positions (uniformly distributed in 3D space)
        self.uav_positions = np.random.uniform(0, [X, Y, H], (N, 3))  # [x, y, z] for each UAV

        # Observation space: positions of BS and UAVs, and current connections
        # BS positions (M*3), UAV positions (N*3), connection matrix (M*N), power allocation (M*N)
        obs_dim = M*3 + N*3 + M*N + M*N
        self._observation_space = Box(low=0, high=max(X, Y, H), shape=(obs_dim,))

        # Action space: connection decisions (M*N) + power allocations (M*N), all in [0,1]
        action_dim = 2 * M * N
        self._action_space = Box(low=0, high=1, shape=(action_dim,))

        self._steps_per_episode = STEPS_PER_EPISODE

        # Initialize connection and power matrices
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        # State includes: BS positions, UAV positions, connection matrix, power matrix
        bs_flat = self.bs_positions.flatten()
        uav_flat = self.uav_positions.flatten()
        conn_flat = self.connection_matrix.flatten()
        power_flat = self.power_matrix.flatten()
        return np.concatenate([bs_flat, uav_flat, conn_flat, power_flat])

    def step(self, action):
        assert not self._terminated, "Episode has terminated"

        # Parse action: first M*N for connections, next M*N for power allocations
        connection_actions = action[:M*N].reshape(M, N)
        power_actions = action[M*N:].reshape(M, N)

        # Update connection matrix: >0.8 means connected
        self.connection_matrix = (connection_actions > 0.8).astype(float)

        # Update power matrix: actual power = P * power_actions
        self.power_matrix = P * power_actions

        # Calculate reward
        reward = self._calculate_reward()

        # Update UAV positions (simple random movement for simulation)
        self._update_uav_positions()

        self._num_steps += 1
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True

        terminated = False
        truncated = self._terminated
        info = {'num_steps': self._num_steps}

        return self.state, reward, terminated, truncated, info

    def reset(self):
        self._num_steps = 0
        self._terminated = False

        # Reinitialize positions
        self.bs_positions = np.random.uniform(0, [X, Y, 0], (M, 3))
        self.uav_positions = np.random.uniform(0, [X, Y, H], (N, 3))
        self.connection_matrix = np.zeros((M, N))
        self.power_matrix = np.zeros((M, N))

        return self.state, {'num_steps': self._num_steps}

    def _calculate_reward(self):
        # Temporary: return fixed reward
        # In future, implement actual reward based on throughput, interference, etc.
        return 1.0

    def _update_uav_positions(self):
        # Simple random movement for UAVs
        movement = np.random.normal(0, 10, (N, 3))  # Small random movements
        self.uav_positions += movement
        # Keep within bounds
        self.uav_positions = np.clip(self.uav_positions, 0, [X, Y, H])

    def seed(self, seed=None):
        np.random.seed(seed)


def make_cellfree_env(training_num=0, test_num=0):
    """Wrapper function for Cell-free UAV env.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = CellFreeEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: CellFreeEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: CellFreeEnv() for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs