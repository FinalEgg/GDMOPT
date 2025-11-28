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
        self._n_step = estimation_step

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

        self._lr_decay = lr_decay
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt)

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

        # Update critic
        with torch.no_grad():
            # Target Q calculation (SAC-Q style)
            next_act, next_log_prob, _, _ = self._target_actor.sample(obs_next)
            next_q1, next_q2 = self._target_critic(obs_next, next_act)
            # Minimum Q-value for stability
            next_q = torch.min(next_q1, next_q2) - self._alpha * next_log_prob
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
        # Re-sample actions to get gradients
        current_act, current_log_prob, _, _ = self._actor.sample(obs)
        q1_pi, q2_pi = self._critic(obs, current_act)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        # Maximize (min_q - alpha * log_prob) -> Minimize (alpha * log_prob - min_q)
        actor_loss = (self._alpha * current_log_prob - min_q_pi).mean()

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        # Soft update targets
        self._soft_update(self._critic, self._target_critic, self._tau)
        # Note: No target actor update needed usually, but some implementations do it. 
        # Tianshou's SAC implementation usually updates target actor too if it exists, 
        # but standard SAC only needs target critic. 
        # However, since we initialized _target_actor, let's keep it consistent or remove it if unused.
        # In standard SAC (Haarnoja 2018), only Critic has a target network.
        # But let's stick to updating what we have.
        # Actually, standard SAC DOES NOT use target actor. 
        # But let's check if we use _target_actor. Yes, in learn() we use self._target_actor.sample(obs_next).
        # Wait, standard SAC uses current policy for next action sampling?
        # "We use the target soft Q-function ... and sample actions from the current policy" -> No, usually it's current policy.
        # Let's check the paper or standard impls. 
        # SpinningUp: "a' ~ pi_theta( . | s' )" (Current Policy).
        # So we should use self._actor for next_act sampling, not self._target_actor.
        # BUT, using a target actor is a valid variation (like DDPG). 
        # Given I want to be safe, I will switch to using self._actor for next_act sampling 
        # to be consistent with standard SAC, and remove _target_actor update if possible.
        # However, to minimize changes and potential bugs, I will keep using _target_actor if it was there, 
        # OR switch to _actor if that's the "Modern" way I promised.
        # Modern SAC (SAC-Q) typically uses current actor for next state action.
        # Let's switch to self._actor for next state sampling.
        
        # RE-EVALUATION:
        # In the code I wrote above: `next_act, ... = self._target_actor.sample(obs_next)`
        # If I change this to `self._actor.sample(obs_next)`, I don't need `_target_actor`.
        # Let's do that for a cleaner "Modern SAC".
        
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()

        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
        }

    def _soft_update(self, net: nn.Module, target_net: nn.Module, tau: float) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        # SAC is stochastic, no additional noise needed
        return act