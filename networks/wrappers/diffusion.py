import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiffusionWrapper(nn.Module):
    """
    Diffusion Policy Wrapper (DDPM).
    扩散策略包装器，基于 DDPM (Denoising Diffusion Probabilistic Models)。
    
    Function (功能):
    - Converts a noise prediction network into a generative policy.
    - 将噪声预测网络转化为生成式策略。
    
    Processes (过程):
    1. Forward (Inference/Sampling): Reverse diffusion process (Denoising) to generate actions.
       前向推理 (采样): 逆向扩散过程 (去噪)，从高斯噪声生成动作。
    2. Loss (Training): Forward diffusion (Add Noise) and compute MSE loss of noise prediction.
       损失计算 (训练): 前向扩散过程 (加噪)，计算预测噪声与真实噪声的均方误差。
    
    Backbone Assumption (骨干网络假设):
    - The `model` must accept concatenated input: [State, Action_Noisy, Time_Normalized].
    - `model` output: Predicted Noise (same shape as Action).
    """
    def __init__(self, 
                 model: nn.Module, 
                 state_dim: int, 
                 action_dim: int, 
                 max_action: float,
                 beta_schedule='linear', 
                 n_timesteps=100
                 ):
        """
        Initialize Diffusion Wrapper.
        
        Args:
            model (nn.Module): Noise prediction network (epsilon_theta).
            state_dim (int): State dimension.
            action_dim (int): Action dimension.
            max_action (float): Action scaling bound.
            beta_schedule (str): 'linear' or 'cosine' noise schedule.
            n_timesteps (int): Number of diffusion steps (T).
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model # The backbone predicting noise eps_theta(s, a, t)
        
        # Diffusion Parameters (扩散参数)
        self.n_timesteps = int(n_timesteps)
        
        if beta_schedule == 'linear':
            # Linear schedule from 1e-4 to 2e-2
            betas = torch.linspace(1e-4, 2e-2, self.n_timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule (Improved DDPM)
            x = torch.linspace(0, self.n_timesteps, self.n_timesteps + 1)
            alphas_cumprod = torch.cos(((x / self.n_timesteps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers (saved in state_dict but not optimized) (注册缓冲区)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def forward(self, state, *args, **kwargs):
        """
        Actor Inference: Generate Action via Sampling.
        策略推理：通过采样生成动作。
        
        Algorithm (算法):
        - Start with random noise x_T ~ N(0, I).
        - Iteratively denoise: x_{t-1} = 1/sqrt(alpha) * (x_t - gamma * eps_theta) + sigma * z.
        
        Returns:
            (Action, None): Action tensor and dummy state for Tianshou compatibility.
        """
        # Ensure state is torch tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.betas.device)
            
        batch_size = state.shape[0]
        
        # 1. Start from pure noise (从纯噪声开始)
        x = torch.randn((batch_size, self.action_dim), device=state.device)
        
        # 2. Denoising Loop (去噪循环)
        for i in reversed(range(self.n_timesteps)):
            # Time embedding (Simplified as normalized float)
            t = torch.full((batch_size,), i, device=state.device, dtype=torch.long)
            t_float = t.float().unsqueeze(1) / self.n_timesteps # Normalize t to [0, 1]
            
            # Prepare Input (State + NoisyAction + Time)
            if state.dim() > 2: state_flat = state.reshape(batch_size, -1)
            else: state_flat = state
            
            # Concat simple vector
            model_input = torch.cat([state_flat, x, t_float], dim=1)
            
            # Predict Noise
            estimated_noise = self.model(model_input)
            
            # Update x (DDPM sampling step)
            alpha = 1 - self.betas[i]
            alpha_bar = self.alphas_cumprod[i]
            sigma = torch.sqrt(self.betas[i]) if i > 0 else 0
            
            # x_{t-1} = ...
            # coeff = beta / sqrt(1 - alpha_bar)
            coeff = self.betas[i] / torch.sqrt(1 - self.alphas_cumprod[i])
            
            mean = (1 / torch.sqrt(alpha)) * (x - coeff * estimated_noise)
            
            # Add noise (Langevin dynamics stochasticity) except for last step
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + sigma * noise
            else:
                x = mean
           
        # 3. Clip Action (裁剪动作)
        action = torch.clamp(x, -self.max_action, self.max_action)
        return action, None # State placeholder

    def loss(self, state, action):
        """
        Compute Diffusion Training Loss.
        计算扩散训练损失。
        
        Algorithm:
        - Sample random time step t.
        - Add noise to action x_0 to get x_t.
        - Predict noise using model.
        - Minimize MSE(PredictedNoise, TrueNoise).
        """
        if not isinstance(state, torch.Tensor):
             state = torch.tensor(state, dtype=torch.float32, device=self.betas.device)
        if not isinstance(action, torch.Tensor):
             action = torch.tensor(action, dtype=torch.float32, device=self.betas.device)

        batch_size = state.shape[0]
        
        # 1. Sample Time t (随机采样时间步)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=state.device).long()
        
        # 2. Sample Noise epsilon (随机采样噪声)
        noise = torch.randn_like(action)
        
        # 3. Forward Diffusion: Get x_t (获取加噪后的 x_t)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        a_bar = self.alphas_cumprod[t].unsqueeze(1) # (B, 1)
        # Fix shape mismatch if necessary
        x_t = torch.sqrt(a_bar) * action + torch.sqrt(1 - a_bar) * noise
        
        # 4. Neural Net Prediction (模型预测)
        t_float = t.float().unsqueeze(1) / self.n_timesteps
        if state.dim() > 2: state_flat = state.reshape(batch_size, -1)
        else: state_flat = state
        
        model_input = torch.cat([state_flat, x_t, t_float], dim=1)
        
        estimated_noise = self.model(model_input)
        
        # 5. MSE Loss (均方误差)
        loss = F.mse_loss(estimated_noise, noise)
        return loss
