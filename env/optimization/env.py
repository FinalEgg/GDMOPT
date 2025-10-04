import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class OptimizationEnv(gym.Env):
    def __init__(self, dim=2, max_steps=100):
        self.dim = dim
        self.max_steps = max_steps
        self.current_step = 0

        # State: current position vector
        self.observation_space = Box(low=-10, high=10, shape=(dim,), dtype=np.float32)

        # Action: movement vector
        self.action_space = Box(low=-1, high=1, shape=(dim,), dtype=np.float32)

        self.reset()

    def step(self, action):
        self.state += action  # Move
        self.state = np.clip(self.state, -10, 10)  # Clip to bounds
        self.current_step += 1

        # Reward: negative of quadratic function f(x) = sum(x_i^2)
        reward = -np.sum(self.state ** 2)

        terminated = False
        truncated = self.current_step >= self.max_steps

        return self.state.copy(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-5, 5, self.dim)
        self.current_step = 0
        return self.state.copy(), {}

def make_optimization_env(training_num=0, test_num=0, dim=2):
    """Wrapper function for Optimization env.
    :return: a tuple of (single env, training envs, test envs).
    """
    from tianshou.env import DummyVectorEnv
    env = OptimizationEnv(dim=dim)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: OptimizationEnv(dim=dim) for _ in range(training_num)])

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: OptimizationEnv(dim=dim) for _ in range(test_num)])
    return env, train_envs, test_envs