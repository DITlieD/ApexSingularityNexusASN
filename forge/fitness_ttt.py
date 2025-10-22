import sys
import os
import numpy as np
import polars as pl
from deap import gp

from serialize_strategy import deap_to_json

# --- Setup Nexus Connection ---
# ... [Keep Nexus import logic from previous steps] ...
# Ensure 'nexus' is imported.
# --- Add the Rust module to the Python path ---
# Check both release and debug paths
module_path_release = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "release")
module_path_debug = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "debug")

if os.path.exists(module_path_release):
    sys.path.append(module_path_release)
elif os.path.exists(module_path_debug):
    sys.path.append(module_path_debug)
else:
    # Allow continuation if nexus isn't built, but evaluation will fail.
    print(f"WARNING: Rust 'nexus' module path not found.")


try:
    import nexus
except ImportError as e:
    print(f"Failed to import 'nexus' module: {e}")
    # Allow continuation for framework testing


# --- Configuration ---
SIM_CONFIG = None
BACKTEST_CONFIG = None # NEW

if 'nexus' in sys.modules:
    SIM_CONFIG = nexus.SimulationConfig(
        num_ticks=1000, taker_fee_rate=0.00055, maker_fee_rate=0.0002,
        initial_capital=200.0, target_capital_factor=2.0
    )
    # NEW: Causal Backtest Config
    BACKTEST_CONFIG = nexus.BacktestConfig(
        initial_capital=200.0, target_capital_factor=2.0,
        fee_rate=0.00055, slippage_factor=0.0001
    )

# Define weights for the Dual-Fitness function (ACN Synthesis)
ADVERSARIAL_WEIGHT = 0.4
CAUSAL_WEIGHT = 0.6 # Prioritize Causal fitness (Robustness)

# --- The Dual-Fitness Evaluation Function ---

# Update signature to accept historical data and feature names
def evaluate_individual(individual, dsg, historical_data_pl: pl.DataFrame=None, feature_names: list[str]=None, onnx_path: str=None):
    """
    Evaluates a GP individual using the Dual-Fitness function (ACN Synthesis).
    """
    if SIM_CONFIG is None or BACKTEST_CONFIG is None:
        return (-float('inf'),)

    # Ensure historical data is provided for Causal fitness
    if historical_data_pl is None or feature_names is None:
         # Fallback mode if data is missing
         return (evaluate_adversarial(individual, dsg),)

    # 1. Serialize the individual
    try:
        strategy_json = deap_to_json(individual, indent=None)
    except Exception as e:
        return (-float('inf'),)

    # 2. Calculate Adversarial Fitness (HF-ABM)
    adversarial_fitness = evaluate_adversarial(individual, dsg, strategy_json=strategy_json, onnx_path=onnx_path)

    # 3. Calculate Causal Fitness (Vectorized Backtest)
    try:
        # Pass the Polars DataFrame directly to the Rust backtester
        causal_fitness = nexus.run_vectorized_backtest_py(
            strategy_json,
            historical_data_pl,
            BACKTEST_CONFIG,
            feature_names
        )
    except Exception as e:
        causal_fitness = -float('inf')

    # 4. Combine Fitness (Weighted Average)
    if adversarial_fitness == -float('inf') or causal_fitness == -float('inf'):
        return (-float('inf'),)

    combined_fitness = (adversarial_fitness * ADVERSARIAL_WEIGHT) + (causal_fitness * CAUSAL_WEIGHT)

    return (combined_fitness,)

def evaluate_adversarial(individual, dsg, strategy_json=None, onnx_path=None):
    """Helper for calculating only adversarial fitness."""
    if strategy_json is None:
        try:
            strategy_json = deap_to_json(individual, indent=None)
        except:
            return -float('inf')
            
    try:
        # Updated call to the Rust runner
        fitness = nexus.run_accelerated_simulation_py(
            SIM_CONFIG,
            strategy_json,
            onnx_path, # Pass the ONNX path
            dsg
        )
        return fitness
    except Exception as e:
        return -float('inf')