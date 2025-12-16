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
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 0.99,
            alpha: Union[float, tuple] = 0.2,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            sparsity_coef: float = 0.01, # Sparsity regularization coefficient
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        self._device = device
        self._tau = tau
        self._gamma = gamma
        self._action_dim = action_dim
        self._n_step = estimation_step
        self.sparsity_coef = sparsity_coef # Store sparsity coef

        # Auto-Alpha Logic
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            if len(alpha) == 4:
                self._target_entropy, self._log_alpha_optim, self._alpha_lr, initial_alpha = alpha
            else:
                self._target_entropy, self._log_alpha_optim, self._alpha_lr = alpha
                initial_alpha = 1.0
            
            self._log_alpha = torch.tensor([np.log(initial_alpha)], requires_grad=True, device=device, dtype=torch.float32)
            self._alpha = self._log_alpha.exp().item()
            # Re-create optimizer for log_alpha
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._alpha_lr)
        else:
            self._alpha = alpha

        # Initialize actor
        if actor is not None:
            self._actor: torch.nn.Module = actor
            # Standard SAC uses current actor for next state sampling, no target actor needed
            if actor_optim is not None:
                self._actor_optim: torch.optim.Optimizer = actor_optim

        # Initialize critic
        if critic is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            if critic_optim is not None:
                self._critic_optim: torch.optim.Optimizer = critic_optim

        self._lr_decay = lr_decay
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt)
            if self._is_auto_alpha:
                 self._alpha_lr_scheduler = CosineAnnealingLR(self._alpha_optim, T_max=lr_maxt)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs: Any) -> Batch:
        obs = batch.obs
        obs = to_torch(obs, device=self._device, dtype=torch.float32)
        # Unpack 5 values
        actions, log_probs, _, _, _ = self._actor.sample(obs)
        return Batch(act=actions, log_prob=log_probs)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # Convert to torch
        obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        act = to_torch(batch.act, device=self._device, dtype=torch.float32)
        rew = to_torch(batch.rew, device=self._device, dtype=torch.float32)
        obs_next = to_torch(batch.obs_next, device=self._device, dtype=torch.float32)
        done = to_torch(batch.done, device=self._device, dtype=torch.float32)

        # Update critic
        with torch.no_grad():
            # Target Q calculation (SAC-Q style)
            # Use current actor for next state action sampling (Standard SAC)
            # Unpack 5 values
            next_act, next_log_prob, _, _, _ = self._actor.sample(obs_next)
            next_q1, next_q2 = self._target_critic(obs_next, next_act)
            # Minimum Q-value for stability
            # Use current alpha
            alpha = self._log_alpha.exp() if self._is_auto_alpha else self._alpha
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob
            # Bellman target
            target_q = rew + (1 - done) * (self._gamma ** self._n_step) * next_q

        # Current Q estimates
        q1, q2 = self._critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        # Gradient Clipping for Critic
        nn.utils.clip_grad_norm_(self._critic.parameters(), max_norm=1.0)
        self._critic_optim.step()

        # Update actor
        actor_loss_item = 0.0
        alpha_loss_item = 0.0
        
        # Check if we should update actor (default True)
        update_actor = kwargs.get("update_actor", True)
        
        if update_actor:
            # Re-sample actions to get gradients
            # Unpack 5 values: action, log_prob, mean, log_std, gate_probs
            current_act, current_log_prob, _, _, gate_probs = self._actor.sample(obs)
            q1_pi, q2_pi = self._critic(obs, current_act)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # Use current alpha
            alpha = self._log_alpha.exp() if self._is_auto_alpha else self._alpha
            
            # Maximize (min_q - alpha * log_prob) -> Minimize (alpha * log_prob - min_q)
            # Add Sparsity Loss: Minimize mean(gate_probs)
            sparsity_loss = self.sparsity_coef * gate_probs.mean()
            actor_loss = (alpha * current_log_prob - min_q_pi).mean() + sparsity_loss

            self._actor_optim.zero_grad()
            actor_loss.backward()
            self._actor_optim.step()
            actor_loss_item = actor_loss.item()
            
            # Update Alpha (Auto-Tuning)
            if self._is_auto_alpha:
                # Loss = - (log_alpha * (log_prob + target_entropy)).mean()
                # We want alpha * log_prob = alpha * (-target_entropy)
                # So log_prob should be close to -target_entropy
                alpha_loss = -(self._log_alpha * (current_log_prob + self._target_entropy).detach()).mean()
                
                self._alpha_optim.zero_grad()
                alpha_loss.backward()
                self._alpha_optim.step()
                alpha_loss_item = alpha_loss.item()
                
                # Update self._alpha for logging/next step usage (though we use .exp() directly)
                self._alpha = self._log_alpha.exp().item()

        # Soft update targets
        self._soft_update(self._critic, self._target_critic, self._tau)
        
        if self._lr_decay:
            if update_actor:
                self._actor_lr_scheduler.step()
                if self._is_auto_alpha:
                    self._alpha_lr_scheduler.step()
            self._critic_lr_scheduler.step()

        return {
            "loss/actor": actor_loss_item,
            "loss/critic": critic_loss.item(),
            "loss/alpha": alpha_loss_item,
            "alpha": self._alpha
        }

    def _soft_update(self, net: nn.Module, target_net: nn.Module, tau: float) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        # SAC is stochastic, no additional noise needed
        return act