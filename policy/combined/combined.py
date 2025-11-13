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
from .helpers import (
    Losses
)

class CombinedOPT(BasePolicy):

    def __init__(
            self,
            state_dim: int,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            connection_dim: int,
            power_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 1,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            bc_coef: bool = False,
            exploration_noise: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        # Initialize actor network and optimizer if provided
        if actor is not None:
            self._actor: torch.nn.Module = actor  # Combined model
            self._target_actor = deepcopy(actor)  # Target actor network for stable learning
            self._target_actor.eval()  # Set target actor to evaluation mode
            self._actor_optim: torch.optim.Optimizer = actor_optim  # Optimizer for the actor network
            self._connection_dim = connection_dim
            self._power_dim = power_dim
            self._action_dim = connection_dim + power_dim  # Total action dimension

        # Initialize critic network and optimizer if provided
        if critic is not None:
            self._critic: torch.nn.Module = critic  # Critic network
            self._target_critic = deepcopy(critic)  # Target critic network
            self._target_critic.eval()  # Set target critic to evaluation mode
            self._critic_optim: torch.optim.Optimizer = critic_optim  # Optimizer for the critic network

        self._device = device
        self._tau = tau
        self._gamma = gamma
        self._reward_normalization = reward_normalization
        self._estimation_step = estimation_step
        self._lr_decay = lr_decay
        self._lr_maxt = lr_maxt
        self._bc_coef = bc_coef
        self._exploration_noise = exploration_noise

        # Learning rate scheduler for actor if lr_decay is enabled
        if self._lr_decay:
            self._actor_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt)

        # Learning rate scheduler for critic if lr_decay is enabled
        if self._lr_decay:
            self._critic_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt)

    def set_exp_noise(self, noise: Union[torch.Tensor, float, np.ndarray]) -> None:
        """Set the exploration noise."""
        self._exploration_noise = noise

    def train(self, mode: bool = True) -> "CombinedOPT":
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
        """Compute the target Q value."""
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next = torch.as_tensor(batch.obs_next, device=self._device, dtype=torch.float)
        with torch.no_grad():
            # Use target actor to get next actions
            connection_next, power_next = self._target_actor(obs_next)
            act_next = torch.cat([connection_next, power_next], dim=1)
            target_q = self._target_critic.q_min(obs_next, act_next)
        return target_q

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer."""
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
        obs = batch.obs
        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float)
        # Get actions from actor
        connection, power = self._actor(obs)
        act = torch.cat([connection, power], dim=1)
        return Batch(act=act)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Update the policy network and replay buffer."""
        # Convert to tensor
        obs = batch.obs
        act = batch.act
        obs_next = batch.obs_next
        rew = batch.rew
        done = batch.done

        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float)
        act = torch.as_tensor(act, device=self._device, dtype=torch.float)
        obs_next = torch.as_tensor(obs_next, device=self._device, dtype=torch.float)
        rew = torch.as_tensor(rew, device=self._device, dtype=torch.float).unsqueeze(1)
        done = torch.as_tensor(done, device=self._device, dtype=torch.float).unsqueeze(1)

        # Split actions into connection and power
        connection_act = act[:, :self._connection_dim]
        power_act = act[:, self._connection_dim:]

        # Critic loss
        q_value = self._critic.q_min(obs, act)
        with torch.no_grad():
            # Target actions
            connection_next, power_next = self._target_actor(obs_next)
            act_next = torch.cat([connection_next, power_next], dim=1)
            target_q = self._target_critic.q_min(obs_next, act_next)
            target_q = rew + (1 - done) * self._gamma * target_q

        critic_loss = F.mse_loss(q_value, target_q)

        # Actor loss
        connection_pred, power_pred = self._actor(obs)
        act_pred = torch.cat([connection_pred, power_pred], dim=1)
        actor_loss = -self._critic.q_min(obs, act_pred).mean()

        # Update critic
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        # Update actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        # Update target networks
        self.sync_weight()

        # Learning rate decay
        if self._lr_decay:
            self._actor_scheduler.step()
            self._critic_scheduler.step()

        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        """Add exploration noise to action."""
        if isinstance(act, np.ndarray) and not np.isinf(act).any():
            act += np.random.normal(0, self._exploration_noise, size=act.shape)
            act = np.clip(act, 0, 1)  # Assuming actions are in [0,1]
        return act