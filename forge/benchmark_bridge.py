import time
import os
import sys
import numpy as np
import polars as pl

# --- Add the Rust module to the Python path ---
# This boilerplate allows the script to find the compiled Rust library
module_path_release = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "release")
module_path_debug = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "debug")

if os.path.exists(module_path_release):
    sys.path.append(module_path_release)
elif os.path.exists(module_path_debug):
    sys.path.append(module_path_debug)
else:
    print("ERROR: Rust 'nexus' module not found. Please compile the Rust project first.")
    sys.exit(1)

try:
    import nexus
except ImportError as e:
    print(f"Failed to import 'nexus' module: {e}")
    sys.exit(1)

def benchmark_bridge_latency(n_iterations=1000):
    """
    Benchmarks the latency of calling a Rust function from Python via PyO3.
    """
    print("--- Benchmarking Python-to-Rust Bridge Latency ---")
    
    # 1. Prepare inputs for the Rust function. We'll use the signal backtester.
    # Create a realistic Polars DataFrame
    data = pl.DataFrame({
        'gp_signal': np.random.uniform(-1.0, 1.0, size=1000),
        'gp_size': np.random.uniform(0.01, 0.5, size=1000),
        'target_return': np.random.normal(0.0, 0.001, size=1000),
        'close': np.random.uniform(20000, 70000, size=1000),
    }).with_columns(pl.all().cast(pl.Decimal(scale=8, precision=None)))

    # Create a backtest configuration object
    config = nexus.BacktestConfig(
        initial_capital=200.0,
        target_capital_factor=2.0,
        fee_rate=0.00055,
        slippage_factor=0.0001
    )

    latencies_ns = []

    # 2. Run the benchmark loop
    print(f"Running {n_iterations} iterations...")
    for _ in range(n_iterations):
        start_time = time.perf_counter_ns()
        
        # Call the Rust function
        _ = nexus.run_signal_backtest_py(data, config)
        
        end_time = time.perf_counter_ns()
        latencies_ns.append(end_time - start_time)

    # 3. Report the results
    avg_latency_ns = np.mean(latencies_ns)
    avg_latency_us = avg_latency_ns / 1000
    p95_latency_us = np.percentile(latencies_ns, 95) / 1000
    
    print("\n--- Benchmark Results ---")
    print(f"Function called: nexus.run_signal_backtest_py")
    print(f"Iterations:      {n_iterations}")
    print(f"Average Latency: {avg_latency_us:.2f} µs (microseconds)")
    print(f"95th Percentile: {p95_latency_us:.2f} µs (microseconds)")
    print("-------------------------")

if __name__ == "__main__":
    benchmark_bridge_latency()
