# Model prediction for connection strengths
import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from model.actor import Actor
from model.sac import Actor as SACActor, Critic, Value
from model.diffusion import MLP, DoubleCritic
from policy.ddpg import DDPG
from policy.sac import SAC

def load_model(model_type, weight_path):
    """
    Load a trained model from the given path.

    Args:
        model_type (str): Type of model ('ddpg', 'sac', 'diffusion')
        weight_path (str): Path to the model weights file

    Returns:
        model: Loaded model ready for prediction
    """
    try:
        if model_type == 'ddpg':
            actor = Actor(state_dim=45, action_dim=24)
            critic = DoubleCritic(state_dim=45, action_dim=24)
            actor_optim = torch.optim.Adam(actor.parameters())
            critic_optim = torch.optim.Adam(critic.parameters())

            policy = DDPG(
                state_dim=45,
                actor=actor,
                actor_optim=actor_optim,
                action_dim=24,
                critic=critic,
                critic_optim=critic_optim,
                device='cpu'
            )

            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
            policy.load_state_dict(checkpoint)
            model = policy._actor  # Use the actor part

        elif model_type == 'sac':
            actor = SACActor(state_dim=45, action_dim=24)
            critic = Critic(state_dim=45, action_dim=24)
            value = Value(state_dim=45)
            actor_optim = torch.optim.Adam(actor.parameters())
            critic_optim = torch.optim.Adam(critic.parameters())
            value_optim = torch.optim.Adam(value.parameters())

            policy = SAC(
                state_dim=45,
                actor=actor,
                actor_optim=actor_optim,
                action_dim=24,
                critic=critic,
                critic_optim=critic_optim,
                value=value,
                value_optim=value_optim,
                device='cpu'
            )

            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
            policy.load_state_dict(checkpoint)
            model = policy._actor

        elif model_type == 'diffusion':
            # Diffusion loading is complex, simplified
            model = MLP(state_dim=45, action_dim=24)
            # Would need proper diffusion model loading

        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def model_prediction(model, model_type, state, M=4, N=3):
    """
    Use the trained model to predict connections and power allocations.
    Returns connection strengths for visualization.

    Args:
        model: Trained neural network model
        model_type (str): Type of model ('ddpg', 'sac', 'diffusion')
        state (np.array): Current state vector
        M (int): Number of base stations
        N (int): Number of UAVs

    Returns:
        np.array: Connection strengths (M*N values)
    """
    if model is None:
        return np.zeros(M*N)

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if model_type in ['ddpg', 'sac']:
            output = model(state_tensor)
            # First M*N values are connections, next M*N are power allocations
            connections = output[0][:M*N].numpy()
        elif model_type == 'diffusion':
            # Simplified implementation - would need proper diffusion inference
            connections = np.random.rand(M*N)
        else:
            connections = np.zeros(M*N)

        return connections
