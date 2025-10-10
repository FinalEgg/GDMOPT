import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR

class SAC(BasePolicy):

    def __init__(
            self,
            state_dim: int,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            value: Optional[torch.nn.Module],
            value_optim: Optional[torch.optim.Optimizer],
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: float = 0.2,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        self._device = device
        self._tau = tau
        self._gamma = gamma
        self._alpha = alpha
        self._action_dim = action_dim

        # Initialize actor
        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim

        # Initialize critic
        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        # Initialize value
        if value is not None and value_optim is not None:
            self._value: torch.nn.Module = value
            self._target_value = deepcopy(value)
            self._target_value.eval()
            self._value_optim: torch.optim.Optimizer = value_optim

        self._lr_decay = lr_decay
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt)
            self._value_lr_scheduler = CosineAnnealingLR(self._value_optim, T_max=lr_maxt)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs: Any) -> Batch:
        obs = batch.obs
        obs = to_torch(obs, device=self._device, dtype=torch.float32)
        actions, log_probs, _, _ = self._actor.sample(obs)
        return Batch(act=actions, log_prob=log_probs)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # Convert to torch
        obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        act = to_torch(batch.act, device=self._device, dtype=torch.float32)
        rew = to_torch(batch.rew, device=self._device, dtype=torch.float32)
        obs_next = to_torch(batch.obs_next, device=self._device, dtype=torch.float32)
        done = to_torch(batch.done, device=self._device, dtype=torch.float32)

        # Update value network
        with torch.no_grad():
            next_act, next_log_prob, _, _ = self._target_actor.sample(obs_next)
            next_q1, next_q2 = self._target_critic(obs_next, next_act)
            next_q = torch.min(next_q1, next_q2) - self._alpha * next_log_prob
            target_value = rew + (1 - done) * self._gamma * next_q

        current_value = self._value(obs)
        value_loss = F.mse_loss(current_value, target_value)

        self._value_optim.zero_grad()
        value_loss.backward()
        self._value_optim.step()

        # Update critic
        with torch.no_grad():
            current_act, current_log_prob, _, _ = self._actor.sample(obs)
            current_q1, current_q2 = self._critic(obs, current_act)
            current_q = torch.min(current_q1, current_q2) - self._alpha * current_log_prob
            target_q = rew + (1 - done) * self._gamma * current_q

        q1, q2 = self._critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        # Update actor
        current_act, current_log_prob, _, _ = self._actor.sample(obs)
        q1_pi, q2_pi = self._critic(obs, current_act)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self._alpha * current_log_prob - min_q_pi).mean()

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        # Soft update targets
        self._soft_update(self._critic, self._target_critic, self._tau)
        self._soft_update(self._value, self._target_value, self._tau)

        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
            self._value_lr_scheduler.step()

        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/value": value_loss.item(),
        }

    def _soft_update(self, net: nn.Module, target_net: nn.Module, tau: float) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        # SAC is stochastic, no additional noise needed
        return act