import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class OptimizationEnv(gym.Env):
    """
    Base class for optimization environments.
    State: Current position x (dim,)
    Action: Delta x (dim,) scaled by step_size
    Reward: -f(x) or improvement
    """
    def __init__(self, dim=2, max_steps=100, bounds=10.0):
        self.dim = dim
        self.max_steps = max_steps
        self.bounds = bounds
        self._num_steps = 0
        
        self.observation_space = Box(low=-bounds, high=bounds, shape=(dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
        
        self.state = np.zeros(dim)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._num_steps = 0
        # Random start
        self.state = np.random.uniform(-self.bounds/2, self.bounds/2, self.dim)
        return self.state, {}

    def step(self, action):
        self._num_steps += 1
        
        # Action is delta, scaled
        step_size = 0.5
        delta = action * step_size
        
        self.state = np.clip(self.state + delta, -self.bounds, self.bounds)
        
        reward = self._calculate_reward()
        
        terminated = False
        truncated = self._num_steps >= self.max_steps
        
        return self.state, reward, terminated, truncated, {}

    def _calculate_reward(self):
        raise NotImplementedError

class QuadraticEnv(OptimizationEnv):
    """
    Convex function: f(x) = sum(x^2)
    Global minimum at 0.
    """
    def _calculate_reward(self):
        # Minimize f(x) -> Maximize -f(x)
        val = np.sum(self.state**2)
        return -val

class RastriginEnv(OptimizationEnv):
    """
    Non-convex function: Rastrigin
    f(x) = 10n + sum(x^2 - 10cos(2pi*x))
    Global minimum at 0.
    """
    def _calculate_reward(self):
        A = 10
        n = self.dim
        val = A * n + np.sum(self.state**2 - A * np.cos(2 * np.pi * self.state))
        return -val
