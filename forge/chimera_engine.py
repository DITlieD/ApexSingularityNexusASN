import cma
import numpy as np
import sys
import os
import pandas as pd
import json

# --- Setup Nexus Connection ---
# (Same path logic as fitness_ttt.py)
module_path_release = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "release")
module_path_debug = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "debug")
if os.path.exists(module_path_release): sys.path.append(module_path_release)
elif os.path.exists(module_path_debug): sys.path.append(module_path_debug)

try:
    import nexus
except ImportError as e:
    print(f"Failed to import 'nexus' module: {e}")
    # Allow continuation for analysis testing

# --- Configuration ---
SIM_CONFIG = None
if 'nexus' in sys.modules:
    # Configuration for Chimera runs (Longer duration for stable statistics)
    SIM_CONFIG = nexus.SimulationConfig(
        num_ticks=5000,
        taker_fee_rate=0.00055,
        maker_fee_rate=0.0002,
        initial_capital=200.0, # Base capital (Rust multiplies this for MM)
        target_capital_factor=1.0
    )

# --- Target Statistics (The "Reality" Benchmark) ---

def calculate_target_statistics(filepath="data/historical_mft_data.csv"):
    """
    Loads historical data and calculates the target feature vector.
    """
    # !!! CRITICAL PLACEHOLDER !!!
    # In a production system, load real historical L2 data here and calculate stats.
    
    print("WARNING: Using placeholder target statistics for Chimera Engine.")
    
    # Hypothetical regime. Values are normalized based on expected ranges.
    # [Volatility, Spread, Volume, Trade Count]
    # Normalization factors (adjust based on your market/data):
    MAX_VOL = 0.01
    MAX_SPREAD = 50.0
    MAX_VOLUME = 100000.0
    MAX_COUNT = 5000.0

    return np.array([
        0.008 / MAX_VOL,   # High Volatility
        5.0 / MAX_SPREAD,  # Tight Spread
        70000.0 / MAX_VOLUME, # High Volume
        3000.0 / MAX_COUNT   # Moderate Trade Count
    ])

TARGET_STATS = calculate_target_statistics()

# Normalization factors (must match calculate_target_statistics)
NORM_FACTORS = np.array([0.01, 50.0, 100000.0, 5000.0])

# --- The Objective Function (Microstructure Mimicry) ---

def objective_function(genome):
    """
    Minimizes the distance between the simulation statistics and reality (TARGET_STATS).
    """
    if SIM_CONFIG is None:
        return float('inf')

    # Genome parameters: [order_prob, spread_factor]. 

    # 1. Run the specialized Chimera simulation in Rust
    try:
        sim_stats = nexus.run_accelerated_chimera_simulation_py(
            SIM_CONFIG,
            genome
        )
    except Exception as e:
        # Return high loss if simulation fails
        return float('inf')

    # 2. Extract and normalize the statistics from the Rust result
    sim_vector = np.array([
        sim_stats.realized_volatility,
        sim_stats.avg_spread,
        sim_stats.total_volume,
        sim_stats.trade_count
    ])
    
    normalized_sim_vector = sim_vector / NORM_FACTORS

    # Handle potential NaNs or Infs from simulation
    if np.any(np.isnan(normalized_sim_vector)) or np.any(np.isinf(normalized_sim_vector)):
        return float('inf')

    # 3. Calculate the Euclidean distance (The fitness score to minimize)
    distance = np.linalg.norm(TARGET_STATS - normalized_sim_vector)
    
    return distance

# --- Main Chimera Engine Logic ---

def find_dsg(sigma0=0.3, maxiter=50):
    """
    Runs CMA-ES to find the Dominant Strategy Genome (DSG).
    """
    print("\n--- Chimera Engine Initializing ---")
    print(f"Target Statistics Vector (Normalized): {TARGET_STATS}")
    
    if SIM_CONFIG is None:
        print("ERROR: Nexus module not available. Cannot run Chimera Engine.")
        return None

    # MODIFIED: Initial guess (4 parameters)
    # [order_prob=0.5, spread_factor=0.01, inventory_sensitivity=0.0001, order_size=1.0]
    initial_genome = [0.5, 0.01, 0.0001, 1.0]
    
    # MODIFIED: Define constraints (4 parameters)
    options = {
        # Bounds format: [[min_p1, min_p2, min_p3, min_p4], [max_p1, max_p2, max_p3, max_p4]]
        'bounds': [
            [0.01, 0.0001, 0.00001, 0.1], # Minima: [Prob, Spread, Sensitivity, Size]
            [0.99, 0.05,   0.005,   10.0]  # Maxima
        ], 
        'maxiter': maxiter,
        'tolfun': 1e-5,
        'verbose': -9,
        'verb_disp': 10,
    }

    # Run the optimization
    print(f"Starting CMA-ES optimization (4 parameters)...")
    try:
         best_genome, es = cma.fmin2(objective_function, initial_genome, sigma0, options)
    except Exception as e:
        print(f"\nERROR during CMA-ES optimization. Ensure Rust 'nexus' module is compiled with the latest changes. Details: {e}")
        return None

    
    print("--- Chimera Engine Run Complete ---")
    if best_genome is not None:
        print("Optimal Genome (DSG) found:")
        print(f"  - Order Probability:     {best_genome[0]:.4f}")
        print(f"  - Spread Factor:         {best_genome[1]:.6f}")
        print(f"  - Inventory Sensitivity: {best_genome[2]:.6f}")
        print(f"  - Order Size:            {best_genome[3]:.4f}")
        print(f"  - Final Distance:    {es.result.fbest:.4f}")
        return list(best_genome)
    else:
        print("Optimization failed to converge.")
        return None

if __name__ == "__main__":
    dsg = find_dsg(maxiter=50)
    # When run standalone, save the DSG for the GP framework to use
    if dsg:
        with open("current_dsg.json", "w") as f:
            json.dump({"dsg": dsg}, f)
        print("Saved DSG to 'current_dsg.json'")