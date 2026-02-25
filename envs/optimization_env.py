import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class OptimizationEnv(gym.Env):
    """
    Optimization Benchmark Base Class.
    通用优化基准环境基类。
    
    Logic (逻辑):
    - State: Current parameters 'x' (dim,) (当前解向量).
    - Action: Update direction 'delta_x' (dim,) (更新步长).
    - Reward: Negative objective function value '-f(x)' (目标函数负值).
    - Goal: Find global minimum of f(x) (寻找最小值).
    
    Attributes:
    - dim (int): Dimension of the problem (维度).
    - max_steps (int): Max iterations (最大迭代次数).
    - bounds (float): Search space bounds (搜索空间边界).
    """
    def __init__(self, config):
        if hasattr(config, 'OPT_DIM'):
            self.dim = config.OPT_DIM
            self.max_steps = config.STEPS_PER_EPISODE
            self.bounds = config.OPT_BOUNDS
        else:
            # Fallback defaults if config is missing attributes
            self.dim = 2
            self.max_steps = 100
            self.bounds = 10.0
            
        self._num_steps = 0
        
        self.observation_space = Box(low=-self.bounds, high=self.bounds, shape=(self.dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32)
        
        self.state = np.zeros(self.dim)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._num_steps = 0
        # Random start (随机初始化起点)
        self.state = np.random.uniform(-self.bounds/2, self.bounds/2, self.dim)
        return self.state, {}

    def step(self, action):
        """
        Apply gradient-like update.
        应用类梯度更新。
        """
        self._num_steps += 1
        
        # Action is delta, scaled (动作即为更新增量)
        step_size = 0.5
        delta = action * step_size
        
        # Update state with clipping (更新并裁剪边界)
        self.state = np.clip(self.state + delta, -self.bounds, self.bounds)
        
        # Calculate Reward (Negative Cost)
        reward = self._calculate_reward()
        
        terminated = False
        truncated = self._num_steps >= self.max_steps
        
        return self.state, reward, terminated, truncated, {}

    def _calculate_reward(self):
        """Abstract method for objective function."""
        raise NotImplementedError

class QuadraticEnv(OptimizationEnv):
    """
    Quadratic Convex Function (Sphere Function).
    二次凸函数环境 (球函数)。
    
    f(x) = sum(x^2)
    Global minimum: 0 at x=[0,0,...]
    """
    def _calculate_reward(self):
        # Minimize f(x) -> Maximize -f(x)
        val = np.sum(self.state**2)
        return -val

class RastriginEnv(OptimizationEnv):
    """
    Rastrigin Non-Convex Function.
    Rastrigin 非凸函数环境。
    
    f(x) = 10n + sum(x^2 - 10cos(2pi*x))
    Global minimum: 0 at x=[0,0,...]
    Many local minima (多局部最优).
    """
    def _calculate_reward(self):
        A = 10
        n = self.dim
        val = A * n + np.sum(self.state**2 - A * np.cos(2 * np.pi * self.state))
        return -val
