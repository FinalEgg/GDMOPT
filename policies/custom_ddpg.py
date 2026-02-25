import torch
import torch.nn.functional as F
from tianshou.policy import DDPGPolicy
from tianshou.data import Batch
from typing import Any, Dict, Optional, Union
from .common import SparsityMixin

class CustomDDPGPolicy(DDPGPolicy, SparsityMixin):
    """
    Custom DDPG Policy integrating Sparsity enforcement.
    集成稀疏性约束的自定义 DDPG 策略。

    Inherits from Tianshou's DDPGPolicy and local SparsityMixin.
    继承自 Tianshou 的 DDPGPolicy 和本地的 SparsityMixin。
    """
    
    def __init__(self, *args, sparsity_coef: float = 0.01, **kwargs):
        """
        Initialize CustomDDPGPolicy.
        初始化 CustomDDPGPolicy。

        Args:
        -----
        sparsity_coef : float
            Coefficient for L1 regularization (default: 0.01).
            L1 正则化系数 (默认: 0.01)。
        """
        DDPGPolicy.__init__(self, *args, **kwargs)
        SparsityMixin.__init__(self, sparsity_coef)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """
        Update policy with a given batch of data.
        使用给定的一批数据更新策略。

        Args:
        -----
        batch : Batch
            A batch of data (obs, act, rew, next_obs, done).
            一批数据 (观测, 动作, 奖励, 下一观测, 完成标志)。

        Returns:
        --------
        Dict[str, float]
            A dictionary of training metrics.
            训练指标字典。
        """
        device = next(self.actor.parameters()).device
        
        # Ensure batch is on device
        # 确保 batch 在设备上
        if hasattr(batch, "to_torch"):
             batch.to_torch(dtype=torch.float32, device=device)
        
        # 1. Target Q calculation
        # 1. 目标 Q 值计算
        with torch.no_grad():
            target_act = self.actor_old(batch.obs_next)[0]
            target_q = self.critic_old(batch.obs_next, target_act)
            if hasattr(target_q, 'logits'): 
                target_q = target_q.logits 
            target_q = batch.rew.unsqueeze(1) + (1.0 - batch.done.unsqueeze(1)) * self._gamma * target_q

        # 2. Critic Update
        # 2. 评论家更新
        current_q = self.critic(batch.obs, batch.act)
        if hasattr(current_q, 'logits'): 
            current_q = current_q.logits
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # 3. Actor Update
        # 3. 演员更新
        act_new, _ = self.actor(batch.obs)
        actor_loss = -self.critic(batch.obs, act_new).mean()
        
        # Sparsity Loss
        # 稀疏损失
        sparsity_loss = self._get_sparsity_loss(act_new)
        total_actor_loss = actor_loss + sparsity_loss
        
        self.actor_optim.zero_grad()
        total_actor_loss.backward()
        self.actor_optim.step()
        
        self.sync_weight()
        
        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "critic/q_value_mean": current_q.mean().item(),
            "critic/target_q_mean": target_q.mean().item(),
        }
        self._log_sparsity_metrics(result, act_new)
        return result

