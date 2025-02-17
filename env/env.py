import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np
from . import config  as cnf

class AIGCEnv(gym.Env):

    def __init__(self):

        self._flag = 0
        # Define observation space based on the shape of the state
        # num_points = cnf.NUM_A_AP*3+cnf.NUM_G_AP*2+cnf.NUM_USERS*2
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        # Define action space - discrete space with 3 possible actions
        num_links_a=cnf.NUM_A_AP*cnf.NUM_USERS
        num_links_g=cnf.NUM_G_AP*(cnf.NUM_USERS+cnf.NUM_A_AP)
        num_links=num_links_a+num_links_g
        self._action_space = Discrete(num_links+cnf.NUM_AP)
        self._num_steps = 0
        self._terminated = False
        self._laststate = None
        self.last_expert_action = None
        # Define the number of steps per episode
        self._steps_per_episode = 1

    @property
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    @property
    def action_space(self):
        # Return the action space
        return self._action_space

    
    @property
    def state(self):
        # Provide the current state to the agent
        x_a = np.random.uniform(0, cnf.MAX_X, cnf.NUM_A_AP)
        y_a = np.random.uniform(0, cnf.MAX_Y, cnf.NUM_A_AP)
        h_a = np.random.uniform(0, cnf.MAX_H, cnf.NUM_A_AP)
        x_g = np.random.uniform(0, cnf.MAX_X, cnf.NUM_G_AP)
        y_g = np.random.uniform(0, cnf.MAX_Y, cnf.NUM_G_AP)
        x_u = np.random.uniform(0, cnf.MAX_X, cnf.NUM_USERS)
        y_u = np.random.uniform(0, cnf.MAX_Y, cnf.NUM_USERS)

        reward_in = []
        reward_in.append(0)
        states = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u, reward_in])

        self.position = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u]) 
        self._laststate = states
        return states

    def step(self, action):
        # Check if episode has ended
        assert not self._terminated, "One episodic has terminated"
        # Calculate reward based on last state and action taken
        reward, expert_action, sub_expert_action, real_action = CompUtility(self.position, action)

        

        self._laststate[-1] = reward
        self._laststate[0:-1] = self.position
        self._num_steps += 1
        # Check if episode should end based on number of steps taken
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
        # Information about number of steps taken
        info = {'num_steps': self._num_steps, 'expert_action': expert_action, 'sub_expert_action': sub_expert_action}
        return self._laststate, reward, self._terminated, info

    def reset(self):
        # Reset the environment to its initial state
        self._num_steps = 0
        self._terminated = False
        state = self.state
        return state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        # Set seed for random number generation
        np.random.seed(seed)


def make_aigc_env(training_num=0, test_num=0):
    """Wrapper function for AIGC env.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        # Create multiple instances of the environment for training
        train_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        # Create multiple instances of the environment for testing
        test_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs
