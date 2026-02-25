import torch
import torch.nn.functional as F
import numpy as np
from tianshou.policy import SACPolicy
from tianshou.data import Batch
from typing import Any, Dict, Tuple, Optional, Union
from .common import SparsityMixin

class CustomSACPolicy(SACPolicy, SparsityMixin):
    """
    Custom SAC Policy integrating Sparsity enforcement.
    集成稀疏性约束的自定义 SAC 策略。

    Overrides learn() to add sparsity regularization to actor loss.
    重写 learn() 以将稀疏正则化添加到演员损失中。
    """
    
    def __init__(self, *args, sparsity_coef: float = 0.01, **kwargs):
        """
        Initialize CustomSACPolicy.
        初始化 CustomSACPolicy。

        Args:
        -----
        sparsity_coef : float
            Coefficient for L1 regularization (default: 0.01).
            L1 正则化系数 (默认: 0.01)。
        """
        SACPolicy.__init__(self, *args, **kwargs)
        SparsityMixin.__init__(self, sparsity_coef)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """
        Update policy with a given batch of data.
        使用给定的一批数据更新策略。
        """
        # Tianshou SAC learn + updates
        device = next(self.actor.parameters()).device
        if hasattr(batch, "to_torch"):
             batch.to_torch(dtype=torch.float32, device=device)
        
        obs = batch.obs
        act = batch.act
        rew = batch.rew.unsqueeze(1)
        obs_next = batch.obs_next
        done = batch.done.unsqueeze(1)

        # 1. Critic Update
        # 1. 评论家更新
        with torch.no_grad():
            # Get next action
            act_next_logits, _ = self.actor(obs_next)
            assert isinstance(act_next_logits, tuple)
            dist = torch.distributions.Normal(*act_next_logits)
            u_next = dist.rsample()
            act_next = torch.tanh(u_next)
            log_prob_next = dist.log_prob(u_next).sum(dim=-1, keepdim=True) - \
                            torch.log(1 - act_next.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

            target_q1 = self.critic1_old(obs_next, act_next)
            target_q2 = self.critic2_old(obs_next, act_next)
            target_q = torch.min(target_q1, target_q2) - self._alpha * log_prob_next
            target_q = rew + (1.0 - done) * self._gamma * target_q
        
        # Current Q
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
        # 2. 演员更新
        
        # Freeze critics
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        act_logits, _ = self.actor(obs)
        dist = torch.distributions.Normal(*act_logits)
        u_new = dist.rsample()
        act_new = torch.tanh(u_new)
        log_prob_new = dist.log_prob(u_new).sum(dim=-1, keepdim=True) - \
                       torch.log(1 - act_new.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        current_q1a = self.critic1(obs, act_new)
        current_q2a = self.critic2(obs, act_new)
        current_qa = torch.min(current_q1a, current_q2a)
        
        actor_loss = (self._alpha * log_prob_new - current_qa).mean()

        # Sparsity Loss
        # 稀疏损失
        sparsity_loss = self._get_sparsity_loss(act_new)
        total_actor_loss = actor_loss + sparsity_loss

        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()

        # Unfreeze critics
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # 3. Alpha Update (Automatic Entropy Tuning)
        # 3. Alpha 更新 (自动熵调节)
        if self._is_auto_alpha:
            log_prob_new = log_prob_new.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob_new).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        self.sync_weight()
        
        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "alpha": self._alpha if isinstance(self._alpha, float) else self._alpha.item(),
            "critic/q_value_mean": current_q1.mean().item(),
            "critic/target_q_mean": target_q.mean().item(),
        }
        self._log_sparsity_metrics(result, act_new)
        return result
