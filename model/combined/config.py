# Combined Model Configuration

class CombinedConfig:
    def __init__(self):
        # Actor Network Parameters
        self.actor_hidden_dim = 256
        self.actor_learning_rate = 1e-4
        self.actor_weight_decay = 1e-5
        
        # Diffusion Model Parameters
        self.diffusion_hidden_dim = 256
        self.diffusion_timesteps = 100
        self.diffusion_learning_rate = 1e-4
        self.diffusion_weight_decay = 1e-5
        self.diffusion_beta_schedule = 'linear'
        self.max_power = 1.0
        
        # Threshold Parameters
        self.threshold = 0.5
        self.use_soft_threshold = True
        self.soft_threshold_temperature = 0.1
        
        # Training Parameters
        self.batch_size = 64
        self.num_epochs = 100
        self.gradient_clip = 1.0
        
        # Loss Parameters
        self.connection_loss_weight = 1.0
        self.power_loss_weight = 1.0
        
        # Optimizer Parameters
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")
        return self
    
    def __str__(self):
        """Return string representation of the config"""
        return str(self.__dict__)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return self.__dict__.copy()