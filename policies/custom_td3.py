import torch
import torch.nn.functional as F
import numpy as np
from tianshou.policy import TD3Policy
from tianshou.data import Batch
from typing import Any, Dict, Optional, Union
from .common import SparsityMixin

class CustomTD3Policy(TD3Policy, SparsityMixin):
    """
    Custom TD3 Policy integrating Sparsity enforcement.
    集成稀疏性约束的自定义 TD3 策略。

    Applies sparsity loss during the delayed actor update step.
    在延迟的演员更新步骤中应用稀疏损失。
    """
    
    def __init__(self, *args, sparsity_coef: float = 0.01, **kwargs):
        """
        Initialize CustomTD3Policy.
        初始化 CustomTD3Policy。

        Args:
        -----
        sparsity_coef : float
            Coefficient for L1 regularization (default: 0.01).
            L1 正则化系数 (默认: 0.01)。
        """
        TD3Policy.__init__(self, *args, **kwargs)
        SparsityMixin.__init__(self, sparsity_coef)
        self._last_actor_loss = 0.0
        self._last_sparsity_loss = 0.0
        
        # Ensure update_actor_freq is available (handle Tianshou version differences)
        if not hasattr(self, "_update_actor_freq"):
            self._update_actor_freq = kwargs.get("update_actor_freq", 2)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """
        Update policy with a given batch of data.
        使用给定的一批数据更新策略。
        """
        # Tianshou TD3 learn + Sparsity
        device = next(self.actor.parameters()).device
        if hasattr(batch, "to_torch"):
             batch.to_torch(dtype=torch.float32, device=device)
        
        obs = batch.obs
        act = batch.act
        rew = batch.rew.unsqueeze(1)
        done = batch.done.unsqueeze(1)
        obs_next = batch.obs_next
        
        # 1. Critic Update
        # 1. 评论家更新
        with torch.no_grad():
            target_act = self.actor_old(obs_next)[0]
            # Add noise to target action
            noise = torch.randn_like(target_act) * self._policy_noise
            noise = noise.clamp(-self._noise_clip, self._noise_clip)
            
            # Clamp target action to action space
            target_act = (target_act + noise).clamp(-1.0, 1.0) 
            
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

        # 2. Actor Update (Delayed)
        # 2. 演员更新 (延迟)
        if self._cnt % self._update_actor_freq == 0:
            act_new, _ = self.actor(obs)
            actor_loss = -self.critic1(obs, act_new).mean()
            
            # Sparsity Loss
            # 稀疏损失
            sparsity_loss = self._get_sparsity_loss(act_new)
            total_actor_loss = actor_loss + sparsity_loss
            
            self.actor_optim.zero_grad()
            total_actor_loss.backward()
            self.actor_optim.step()
            
            self.sync_weight()
            
            self._last_actor_loss = actor_loss.item()
            self._last_sparsity_loss = sparsity_loss.item()
        
        result = {
            "loss/actor": self._last_actor_loss,
            "loss/critic": critic_loss.item(),
            "loss/sparsity": self._last_sparsity_loss,
            "critic/q_value_mean": current_q1.mean().item(),
            "critic/target_q_mean": target_q.mean().item(),
        }
        
        # Log sparsity metrics if available from this step, or ignore
        # 如果当前步骤有数据，则记录稀疏性指标
        if self._cnt % self._update_actor_freq == 0:
             # Re-calculate act_new for logging if needed, but we have it inside local scope?
             # Python variables leak scope from if blocks, so act_new is accessible if entered.
             # Python 变量会从 if 块中泄漏，因此如果进入了 if 块，则可以访问 act_new。
             try:
                 # Check if act_new is defined
                 _ = act_new 
                 self._log_sparsity_metrics(result, act_new)
             except UnboundLocalError:
                 pass

        self._cnt += 1
        return result
