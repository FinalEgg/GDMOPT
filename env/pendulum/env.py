import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import math

class PendulumEnv(gym.Env):
    def __init__(self, max_speed=4, max_torque=1, max_force=5, dt=0.02, g=9.8, m=1.0, l=1.0):
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.max_force = max_force
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l

        # State: [x, x_dot, theta, theta_dot]
        self.observation_space = Box(low=np.array([-np.inf, -self.max_speed, -np.pi, -self.max_speed]),
                                     high=np.array([np.inf, self.max_speed, np.pi, self.max_speed]),
                                     shape=(4,), dtype=np.float32)

        # Action: [force_x, torque]
        self.action_space = Box(low=np.array([-self.max_force, -self.max_torque]),
                                high=np.array([self.max_force, self.max_torque]),
                                shape=(2,), dtype=np.float32)

        self.reset()

    def step(self, action):
        force_x, torque = action
        x, x_dot, theta, theta_dot = self.state

        # Physics
        x_acc = force_x / self.m
        theta_acc = (torque / (self.m * self.l**2)) - (self.g / self.l) * np.sin(theta)

        x_dot += x_acc * self.dt
        theta_dot += theta_acc * self.dt
        x += x_dot * self.dt
        theta += theta_dot * self.dt

        # Clip speeds
        x_dot = np.clip(x_dot, -self.max_speed, self.max_speed)
        theta_dot = np.clip(theta_dot, -self.max_speed, self.max_speed)

        self.state = np.array([x, x_dot, theta, theta_dot])

        # Reward: penalize angle deviation and position
        reward = - (theta**2 + 0.1 * theta_dot**2 + 0.01 * x**2 + 0.01 * x_dot**2)

        terminated = False  # No termination
        truncated = False   # No truncation for simplicity

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial state
        theta = np.random.uniform(-np.pi, np.pi)
        theta_dot = np.random.uniform(-1, 1)
        x = np.random.uniform(-0.5, 0.5)
        x_dot = np.random.uniform(-0.5, 0.5)
        self.state = np.array([x, x_dot, theta, theta_dot])
        return self.state, {}

    def render(self):
        # Simple text render
        x, x_dot, theta, theta_dot = self.state
        print(f"State: x={x:.2f}, x_dot={x_dot:.2f}, theta={theta:.2f}, theta_dot={theta_dot:.2f}")

def make_pendulum_env(training_num=0, test_num=0):
    """Wrapper function for Pendulum env.
    :return: a tuple of (single env, training envs, test envs).
    """
    from tianshou.env import DummyVectorEnv
    env = PendulumEnv()
    # env.seed(0)  # Not needed in gymnasium

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: PendulumEnv() for _ in range(training_num)])
        # train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: PendulumEnv() for _ in range(test_num)])
        # test_envs.seed(0)
    return env, train_envs, test_envs