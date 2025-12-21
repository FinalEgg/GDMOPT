import torch
import torch.nn.functional as F
from tianshou.policy import TD3Policy
from tianshou.data import Batch
from typing import Any, Dict, Optional, Union
import numpy as np

class CustomTD3Policy(TD3Policy):
    """
    Custom TD3 Policy with enhanced logging.
    """
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # obs, act, rew, obs_next, done = batch.obs, batch.act, batch.rew, batch.obs_next, batch.done
        # Tianshou batch is already tensors if we use to_torch_batch or similar, 
        # but usually learn receives a Batch object where fields might be numpy.
        # Tianshou's BasePolicy.learn usually handles processing.
        # But since we override, we must handle it.
        
        device = next(self.actor.parameters()).device
        batch.to_torch(dtype=torch.float32, device=device)
        obs = batch.obs
        act = batch.act
        rew = batch.rew.unsqueeze(1)
        done = batch.done.unsqueeze(1)
        obs_next = batch.obs_next
        
        # 1. Critic Update
        with torch.no_grad():
            target_act = self.actor_old(obs_next)[0]
            noise = torch.randn_like(target_act) * self._policy_noise
            noise = noise.clamp(-self._noise_clip, self._noise_clip)
            # Use action_space bounds directly
            low = float(self.action_space.low[0])
            high = float(self.action_space.high[0])
            target_act = (target_act + noise).clamp(low, high)
            
            target_q1 = self.critic1_old(obs_next, target_act)
            target_q2 = self.critic2_old(obs_next, target_act)
            target_q = torch.min(target_q1, target_q2)
            target_q = rew + (1.0 - done) * self._gamma * target_q
            
        current_q1 = self.critic1(obs, act)
        current_q2 = self.critic2(obs, act)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()
        
        # 2. Actor Update
        result = {
            "loss/critic": critic_loss.item(),
            "critic/q_value_mean": current_q1.mean().item(),
            "critic/q_value_std": current_q1.std().item(),
            "critic/target_q_mean": target_q.mean().item(),
            "critic/target_q_std": target_q.std().item(),
        }

        if self._cnt % self._freq == 0:
            act_new = self.actor(obs)[0]
            q1 = self.critic1(obs, act_new)
            actor_loss = -q1.mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            self.sync_weight()
            result["loss/actor"] = actor_loss.item()
            
        self._cnt += 1
        
        # Enhanced Logging
        return result
