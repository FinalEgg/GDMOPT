import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility_gpu import calc_util
import numpy as np
from . import config  as cnf
import time

class AIGCEnv(gym.Env):

    def __init__(self):

        self._flag = 0
        # Define observation space based on the shape of the state
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        # Define action space - discrete space with 3 possible actions
        num_links_a=cnf.NUM_A_AP*cnf.NUM_USERS
        num_links_g=cnf.NUM_G_AP*(cnf.NUM_USERS+cnf.NUM_A_AP)
        num_links=num_links_a+num_links_g
        move = cnf.NUM_A_AP*3
        self._action_space = Box(low=-1, high=1, shape=(move + num_links,), dtype=np.float32)
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
        # 将状态初始化为 [0,1] 之间的小数
        x_a = np.random.uniform(0, 1, cnf.NUM_A_AP)
        y_a = np.random.uniform(0, 1, cnf.NUM_A_AP)
        h_a = np.random.uniform(0, 1, cnf.NUM_A_AP)  # 高度归一化
        x_g = np.random.uniform(0, 1, cnf.NUM_G_AP)
        y_g = np.random.uniform(0, 1, cnf.NUM_G_AP)
        x_u = np.random.uniform(0, 1, cnf.NUM_USERS)
        y_u = np.random.uniform(0, 1, cnf.NUM_USERS)
        num_links_a = cnf.NUM_A_AP * cnf.NUM_USERS
        num_links_g = cnf.NUM_G_AP * (cnf.NUM_USERS + cnf.NUM_A_AP)
        num_links = num_links_a + num_links_g
        p_alloc = np.random.uniform(0, 1, num_links)
    
        reward_in = [0]
        states = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u, p_alloc, reward_in])
    
        self.position = np.concatenate([x_a, y_a, h_a, x_g, y_g, x_u, y_u, p_alloc])
        self._laststate = states
        return states


    def step(self, action):
        # Check if episode has ended
        assert not self._terminated, "One episodic has terminated"
        # Calculate reward based on last state and action taken
        reward, tot_capacity, punishment, link_cost, expert_action, sub_expert_action, real_action = calc_util(self.position, action)
        # action: 3*NUM_A_AP bits for moving, NUM_LINKS bits for power allocation
        start1 = cnf.NUM_A_AP * 3
        start2 = cnf.NUM_A_AP * 3 + cnf.NUM_G_AP * 2 + cnf.NUM_USERS * 2
    
        move_update = self._laststate[:start1] + real_action[:start1]
        n_a = cnf.NUM_A_AP
        # 对各部分位置进行更新，并确保归一化在 [0,1] 内
        move_update[:n_a] = np.clip(move_update[:n_a], 0, 1)
        move_update[n_a:2*n_a] = np.clip(move_update[n_a:2*n_a], 0, 1)
        move_update[2*n_a:3*n_a] = np.clip(move_update[2*n_a:3*n_a], 0, 1)
        self._laststate[:start1] = move_update
    
        # 更新功率分配部分
        self._laststate[start2:-1] = real_action[start1:]
        self._laststate[-1] = reward
        self._num_steps += 1
    
        # 检查是否达到最大步数，从而终止当前回合
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
        info = {
            'num_steps': self._num_steps,
            'expert_action': expert_action,
            'sub_expert_action': sub_expert_action,
            'punishment': punishment,
            'link_cost': link_cost,
            'capacity_sum': tot_capacity
        }
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
    # 使用系统时间生成基种
    base_seed = int(time.time() * 1000) % (2**32 - 1)

    # 单个环境使用基种
    env = AIGCEnv()
    env.seed(base_seed)

    train_envs, test_envs = None, None
    if training_num:
        # 为训练环境生成不同的随机种子
        def make_train_env(i):
            env_i = AIGCEnv()
            env_i.seed(base_seed + i)
            return env_i
        train_envs = DummyVectorEnv([lambda i=i: make_train_env(i) for i in range(training_num)])
    
    if test_num:
        # 为测试环境生成不同的随机种子，与训练环境区分开
        def make_test_env(i):
            env_i = AIGCEnv()
            env_i.seed(base_seed + 10000 + i)
            return env_i
        test_envs = DummyVectorEnv([lambda i=i: make_test_env(i) for i in range(test_num)])
    
    return env, train_envs, test_envs
