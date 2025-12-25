import torch
import torch.nn.functional as F
import numpy as np
from tianshou.policy import SACPolicy
from tianshou.data import Batch
from typing import Any, Dict, Tuple, Optional, Union

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, sparsity_coef=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity_coef = sparsity_coef

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        (mu, sigma), hidden = self.actor(obs, state=state, info=batch.info)
        dist = torch.distributions.Normal(mu, sigma)
        if self._deterministic_eval and not self.training:
            act = torch.tanh(mu)
            log_prob = None
        else:
            u = dist.rsample()
            act = torch.tanh(u)
            # Correct log_prob for Tanh transform
            log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True) - \
                       torch.log(1 - act.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return Batch(logits=(mu, sigma), act=act, state=hidden, dist=dist, log_prob=log_prob)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # Re-implementing learn to capture Q-values
        device = next(self.actor.parameters()).device
        batch.to_torch(dtype=torch.float32, device=device)
        obs = batch.obs
        act = batch.act
        rew = batch.rew.unsqueeze(1)
        obs_next = batch.obs_next
        done = batch.done.unsqueeze(1)

        # 1. Critic Update
        with torch.no_grad():
            # obs_next_result = self.actor(obs_next)
            # act_next = obs_next_result[0]
            # log_prob_next = obs_next_result[1]
            
            # Manual sampling with Tanh squashing (Squashed Gaussian)
            (mu, sigma), _ = self.actor(obs_next)
            dist = torch.distributions.Normal(mu, sigma)
            u_next = dist.rsample()
            act_next = torch.tanh(u_next)
            # Correct log_prob for Tanh transform
            log_prob_next = dist.log_prob(u_next).sum(dim=-1, keepdim=True) - \
                            torch.log(1 - act_next.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            
            target_q1 = self.critic1_old(obs_next, act_next)
            target_q2 = self.critic2_old(obs_next, act_next)
            target_q = torch.min(target_q1, target_q2) - self._alpha * log_prob_next
            target_q = rew + (1.0 - done) * self._gamma * target_q

        current_q1 = self.critic1(obs, act)
        current_q2 = self.critic2(obs, act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        # 2. Actor Update
        # Freeze critic so you don't update it during actor update
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # obs_result = self.actor(obs)
        # act_new = obs_result[0]
        # log_prob = obs_result[1]
        
        (mu, sigma), _ = self.actor(obs)
        dist = torch.distributions.Normal(mu, sigma)
        u_new = dist.rsample()
        act_new = torch.tanh(u_new)
        # Correct log_prob for Tanh transform
        log_prob = dist.log_prob(u_new).sum(dim=-1, keepdim=True) - \
                   torch.log(1 - act_new.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        current_q1a = self.critic1(obs, act_new)
        current_q2a = self.critic2(obs, act_new)
        current_qa = torch.min(current_q1a, current_q2a)
        
        actor_loss = (self._alpha * log_prob - current_qa).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        # 3. Alpha Update
        if self._is_auto_alpha:
            log_prob_obj = log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob_obj).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        self.sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "alpha": self._alpha.item() if self._is_auto_alpha else self._alpha,
            "ent": -log_prob.mean().item(),
            "critic/q_value_mean": current_q1.mean().item(),
            "critic/q_value_std": current_q1.std().item(),
            "critic/target_q_mean": target_q.mean().item(),
            "critic/target_q_std": target_q.std().item(),
        }
