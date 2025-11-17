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

class DDPG(BasePolicy):

    def __init__(
            self,
            state_dim: int,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 0.99,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            exploration_noise: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        # Initialize actor network and optimizer if provided
        if actor is not None:
            self._actor: torch.nn.Module = actor  # Actor network
            self._target_actor = deepcopy(actor)  # Target actor network
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim
            self._action_dim = action_dim

        # Initialize critic network and optimizer if provided
        if critic is not None:
            self._critic: torch.nn.Module = critic  # Critic network
            self._target_critic = deepcopy(critic)  # Target critic network
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        self._device = device
        self._tau = tau
        self._gamma = gamma
        self._reward_normalization = reward_normalization
        self._estimation_step = estimation_step
        self._lr_decay = lr_decay
        self._lr_maxt = lr_maxt
        self._exploration_noise = exploration_noise

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt)

    def set_exp_noise(self, noise: Union[float, torch.Tensor]) -> None:
        """Set the exploration noise."""
        self._exploration_noise = noise

    def train(self, mode: bool = True) -> "DDPG":
        """Set the module in training mode, except for the target networks."""
        self.training = mode
        self._actor.train(mode)
        if hasattr(self, '_critic'):
            self._critic.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the target network."""
        for o, n in zip(self._target_actor.parameters(), self._actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        if hasattr(self, '_target_critic'):
            for o, n in zip(self._target_critic.parameters(), self._critic.parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        obs_next = to_torch(batch.obs_next, device=self._device).float()
        target_act = self._target_actor(obs_next)
        target_q = self._target_critic(obs_next, target_act)
        return torch.min(target_q[0], target_q[1])

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._estimation_step, self._reward_normalization
        )
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data."""
        obs = to_torch(batch.obs, device=self._device).float()
        act = self._actor(obs)
        return Batch(act=act, state=state)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # Convert to tensor
        obs = to_torch(batch.obs, device=self._device).float()
        act = to_torch(batch.act, device=self._device).float()
        
        # Critic update
        current_q1, current_q2 = self._critic(obs, act)
        current_q1 = torch.clamp(current_q1, -100, 100)  # edit: Q-value clipping for regularization
        current_q2 = torch.clamp(current_q2, -100, 100)  # edit: Q-value clipping for regularization
        target_q = batch.returns
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)  # edit: gradient clipping for regularization
        self._critic_optim.step()

        # Actor update
        act_new = self._actor(obs)
        actor_loss = -torch.min(*self._critic(obs, act_new)).mean()  # Use min(q1, q2) for actor update edit

        self._actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 1.0)  # edit: gradient clipping for regularization
        self._actor_optim.step()

        # Soft update
        self.sync_weight()

        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()

        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }