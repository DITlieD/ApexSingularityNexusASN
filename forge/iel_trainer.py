import torch
import torch.nn as nn
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from iel_environment import LOBExecutionEnv, STATE_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IEL_Trainer")

# Configuration
# The ONNX file will be saved in the forge directory (STRATEGY_DIR for Rust).
ONNX_PATH = "iel_model.onnx" 
TRAINING_STEPS = 150000

def train_iel_agent():
    """Trains the PPO agent for the IEL task."""
    logger.info("Initializing IEL Environment and PPO Agent...")
    env = LOBExecutionEnv()

    # Initialize PPO agent with an MLP Policy
    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=[64, 64])
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                learning_rate=0.0003, n_steps=2048, batch_size=64)

    logger.info(f"Starting IEL training for {TRAINING_STEPS} steps...")
    model.learn(total_timesteps=TRAINING_STEPS)

    logger.info("Training finished.")

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    logger.info(f"Trained Agent Mean Reward (Lower Cost): {mean_reward:.6f} +/- {std_reward:.6f}")
    
    return model

# --- ONNX Export ---

class OnnxablePolicy(nn.Module):
    """Wrapper around the SB3 policy for ONNX export."""
    def __init__(self, policy):
        super(OnnxablePolicy, self).__init__()
        # Extract the necessary networks from the SB3 structure (ActorCriticPolicy)
        self.action_net = policy.action_net
        # mlp_extractor holds the feature extraction layers (latent_pi)
        self.latent_pi = policy.mlp_extractor.latent_pi

    def forward(self, observation):
        # Replicate the forward pass: Extract features -> Get action logits
        latent = self.latent_pi(observation)
        return self.action_net(latent)

def export_to_onnx(model):
    """Exports the trained PPO policy to ONNX format."""
    logger.info(f"Exporting IEL model to {ONNX_PATH}...")
    
    # Move policy to CPU and wrap for export
    policy = model.policy.to("cpu")
    onnx_policy = OnnxablePolicy(policy)
    onnx_policy.eval()

    # Create dummy input (Batch=1, State=4)
    dummy_input = torch.randn(1, STATE_DIM)

    try:
        torch.onnx.export(
            onnx_policy,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            # Define names matching the Rust inference expectations
            input_names=['state'],
            output_names=['action_logits']
        )
        logger.info("ONNX export successful.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")

if __name__ == "__main__":
    # Run the training and export process
    trained_model = train_iel_agent()
    if trained_model:
        export_to_onnx(trained_model)