"""
Classic Control Environments Wrapper.
经典控制环境包装器。

Adapts Gymnasium classic control environments to the project's config-driven interface.
将 Gymnasium 经典控制环境适配到本项目的配置驱动接口。
"""

import gymnasium as gym
import numpy as np

class CartPoleEnv(gym.Env):
    """
    Wrapper for CartPole-v1 to accept EnvConfig.
    CartPole-v1 的包装器，用于接收 EnvConfig。
    """
    def __init__(self, config):
        """
        Args:
            config: EnvConfig object.
        """
        self.config = config
        # Use new_step_api for gymnasium compatibility if needed, but Tianshou handles envs well.
        # render_mode=None for faster training
        self._env = gym.make("CartPole-v1", render_mode=None)
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        # Handle discrete action conversion if necessary
        # The project often uses float actions, but CartPole is Discrete(2)
        # If the model outputs float (e.g. from DDPG/SAC), we need to discretize it.
        # However, for now, let's assume the algorithm handles the action space matching
        # OR we add a simple check.
        
        # If action is array (e.g. [0.1]), take argmax or threshold?
        # CartPole: 0 or 1.
        # If continuous input u in [-1, 1], map to 0 or 1.
        if isinstance(self.action_space, gym.spaces.Discrete):
             if isinstance(action, (np.ndarray, list)):
                 if np.size(action) == 1:
                     val = action.item() if isinstance(action, np.ndarray) else action
                     # Threshold at 0
                     action = 1 if val > 0 else 0
                 else:
                     # Softmax or similar logic if dimension > 1
                     action = np.argmax(action)
             elif isinstance(action, (float, np.floating)):
                 action = 1 if action > 0 else 0
                     
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

class PendulumEnv(gym.Env):
    """
    Wrapper for Pendulum-v1 to accept EnvConfig.
    倒立摆环境包装器，用于连续控制基准测试。
    
    Observation:
        Type: Box(3)
        Num	Observation	Min	Max
        0	cos(theta)	-1.0	1.0
        1	sin(theta)	-1.0	1.0
        2	theta dot	-8.0	8.0
        
    Action:
        Type: Box(1)
        Num	Action	Min	Max
        0	Torque	-2.0	2.0
        
    Reward:
        -(theta^2 + 0.1*theta_dot^2 + 0.001*action^2)
        Target is to keep the pendulum upright (theta=0).
    """
    def __init__(self, config):
        self.config = config
        # Pendulum-v1 is standard in Gymnasium
        self._env = gym.make("Pendulum-v1", render_mode=None)
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        # Pendulum expects action in [-2, 2]
        # Our algorithms (SAC/TD3) usually output [-1, 1]
        # So we scale the action.
        
        # Clip to ensure valid range just in case
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to environment bounds
        low = self.action_space.low
        high = self.action_space.high
        
        # Simple linear scaling from [-1, 1] to [low, high]
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        
        return self._env.step(scaled_action)
        
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()

