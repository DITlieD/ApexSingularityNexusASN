import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
import numpy as np
import polars as pl
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAML")

# Configuration (Must match Rust implementation!)
CONTEXT_SIZE = 10 # Sequence length for the GRU context
# Ensure this matches the features available in gp_framework.py and Rust Nexus
FEATURE_SIZE = 5  # e.g., Imbalance, Pressure, CLV, Vol_15m, Mom_15m
MAML_LR = 0.001
ADAPTATION_LR = 0.01
ADAPTATION_STEPS = 5

# --- 1. The Neural Network Modulator ---

class StrategyModulator(nn.Module):
    """
    A GRU network that modulates GP outputs based on recent market context.
    """
    def __init__(self, hidden_size=32):
        super(StrategyModulator, self).__init__()
        self.gru = nn.GRU(input_size=FEATURE_SIZE, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        
        # Initialize weights carefully for stability
        self.fc.weight.data.normal_(0.0, 0.01)
        # Initialize bias so modulation starts near identity.
        # Tanh(0) = 0 (additive identity). Softplus(0.54) ~= 1 (multiplicative identity).
        self.fc.bias.data[0] = 0.0
        self.fc.bias.data[1] = 0.54

    def forward(self, context):
        # context shape: (batch_size, CONTEXT_SIZE, FEATURE_SIZE)
        # We only need the last hidden state from the GRU
        _, hidden = self.gru(context)
        modulation = self.fc(hidden.squeeze(0))
        
        # Signal modulation: Tanh (-1 to 1). Additive factor.
        signal_mod = torch.tanh(modulation[:, 0])
        # Size modulation: Softplus (>0). Multiplicative factor.
        size_mod = torch.nn.functional.softplus(modulation[:, 1])
        return signal_mod, size_mod

# --- 2. MAML Training Manager ---

class MAMLManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = StrategyModulator().to(self.device)
        # Use first_order=True for faster approximation
        self.maml = l2l.algorithms.MAML(self.base_model, lr=ADAPTATION_LR, first_order=True).to(self.device)
        self.optimizer = optim.Adam(self.maml.parameters(), lr=MAML_LR)
        logger.info(f"[MAML] Initialized on {self.device}.")
        
    # Define the loss function (Maximize PnL of the combined strategy)
    def calculate_loss(self, gp_signal, gp_size, signal_mod, size_mod, forward_returns):
        # Apply modulation
        final_signal = gp_signal + signal_mod
        final_size = gp_size * size_mod
        
        # Calculate the realized return of the combined strategy
        # PnL = (Signal * Size) * Forward Return
        # Use Tanh on the final signal to smooth the optimization landscape
        realized_pnl = torch.tanh(final_signal) * final_size * forward_returns
        
        # Loss is negative mean PnL (we want to maximize PnL)
        return -torch.mean(realized_pnl)

    def meta_train(self, task_generator, iterations=500):
        """The MAML training loop (Outer Loop)."""
        logger.info(f"[MAML] Starting meta-training...")
        
        for iteration in range(iterations):
            meta_loss = 0.0
            self.optimizer.zero_grad()

            # Sample a batch of tasks (market scenarios)
            tasks = task_generator.sample_tasks(batch_size=32)
            if not tasks:
                continue

            for task in tasks:
                learner = self.maml.clone()

                # Inner Loop: Adaptation (Support Set)
                s_ctx, s_gp_sig, s_gp_size, s_returns = [t.to(self.device) for t in task['support']]

                for _ in range(ADAPTATION_STEPS):
                    signal_mod, size_mod = learner(s_ctx)
                    adaptation_loss = self.calculate_loss(s_gp_sig, s_gp_size, signal_mod, size_mod, s_returns)
                    learner.adapt(adaptation_loss)

                # Outer Loop: Meta-Update (Query Set)
                q_ctx, q_gp_sig, q_gp_size, q_returns = [t.to(self.device) for t in task['query']]
                
                signal_mod, size_mod = learner(q_ctx)
                validation_loss = self.calculate_loss(q_gp_sig, q_gp_size, signal_mod, size_mod, q_returns)
                meta_loss += validation_loss

            # Update the base model parameters
            if len(tasks) > 0:
                meta_loss = meta_loss / len(tasks)
                meta_loss.backward()
                self.optimizer.step()

                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}: Meta Loss = {meta_loss.item():.6f}")

    def export_onnx(self, path="maml_modulator.onnx"):
        """Exports the trained base model to ONNX format for Rust inference."""
        self.base_model.eval().to("cpu") # Move to CPU for export
        # Dummy input shape: (Batch=1, Seq=10, Features=5)
        dummy_input = torch.randn(1, CONTEXT_SIZE, FEATURE_SIZE)
        try:
            torch.onnx.export(
                self.base_model, dummy_input, path,
                export_params=True, opset_version=11, do_constant_folding=True,
                # Define names matching the Rust inference expectations
                input_names=['context'], output_names=['signal_mod', 'size_mod']
            )
            logger.info(f"[MAML] Model exported to ONNX format at {path}")
        except Exception as e:
            logger.error(f"[MAML] ERROR exporting ONNX: {e}")

# --- 3. Task Generation ---

class TaskGenerator:
    """
    Generates MAML tasks from historical data, including the outputs of the specific GP strategy being modulated.
    """
    def __init__(self, data_pl: pl.DataFrame, gp_outputs: pl.DataFrame, feature_names: list[str], target_name: str):
        if len(feature_names) != FEATURE_SIZE:
             raise ValueError(f"Expected {FEATURE_SIZE} features, got {len(feature_names)}.")
             
        # Ensure data is float32 for PyTorch
        self.features = data_pl.select(feature_names).to_numpy().astype(np.float32)
        
        # Ensure GP outputs are float32 (they might be Decimal from the GP framework)
        self.gp_signal = gp_outputs['gp_signal'].cast(pl.Float32).to_numpy()
        self.gp_size = gp_outputs['gp_size'].cast(pl.Float32).to_numpy()
        self.targets = data_pl[target_name].cast(pl.Float32).to_numpy()
        
        self.task_len = ADAPTATION_STEPS * 2 # Support + Query size
        self.total_len = len(self.features)

    def sample_tasks(self, batch_size):
        tasks = []
        # Ensure enough data for lookback context and the task itself
        if self.total_len < self.task_len + CONTEXT_SIZE:
            logger.warning("[MAML] Not enough data for task generation.")
            return tasks

        for _ in range(batch_size):
            # Sample a starting point (must allow for CONTEXT_SIZE lookback)
            start_idx = np.random.randint(CONTEXT_SIZE, self.total_len - self.task_len)
            
            # Define support and query sets
            support_indices = range(start_idx, start_idx + ADAPTATION_STEPS)
            query_indices = range(start_idx + ADAPTATION_STEPS, start_idx + self.task_len)

            support = self._prepare_tensors(support_indices)
            query = self._prepare_tensors(query_indices)
            
            tasks.append({'support': support, 'query': query})
        return tasks

    def _prepare_tensors(self, indices):
        contexts = []
        for idx in indices:
            # Extract the lookback context (sequence) for the GRU
            context = self.features[idx-CONTEXT_SIZE:idx]
            contexts.append(context)
        
        # Convert to Tensors (Context, GP_Signal, GP_Size, Returns)
        # Context shape: (Batch=Adaptation_Steps, Seq=Context_Size, Features=Feature_Size)
        ctx_tensor = torch.tensor(np.array(contexts))
        gp_sig_tensor = torch.tensor(self.gp_signal[indices])
        gp_size_tensor = torch.tensor(self.gp_size[indices])
        returns_tensor = torch.tensor(self.targets[indices])
        
        return ctx_tensor, gp_sig_tensor, gp_size_tensor, returns_tensor