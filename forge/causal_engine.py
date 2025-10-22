import pandas as pd
import numpy as np

# Tigramite imports
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
except ImportError:
    print("WARNING: Tigramite not installed. Causal analysis unavailable. (pip install tigramite)")
    PCMCI = None

def perform_causal_discovery(df: pd.DataFrame, target_variable: str, tau_max=5, pc_alpha=0.01):
    """
    Performs time-series causal discovery using the PCMCI algorithm.
    """
    if PCMCI is None:
        # Fallback: return all features if Tigramite isn't available
        return list(df.columns.drop(target_variable))

    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataframe.")

    # PCMCI often performs better on standardized data.
    df_std = (df - df.mean()) / df.std()
    df_std = df_std.fillna(0) # Handle cases where std is 0

    print(f"\nStarting Causal Discovery (Target: {target_variable}, TauMax: {tau_max}, Alpha: {pc_alpha})...")
    
    # 1. Prepare data for Tigramite
    var_names = list(df_std.columns)
    dataframe = pp.DataFrame(df_std.values, var_names=var_names)
    
    # 2. Initialize PCMCI (Using Partial Correlation)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    
    # 3. Run the PCMCI algorithm
    try:
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    except Exception as e:
        print(f"ERROR during PCMCI execution: {e}. Falling back to all features.")
        return list(df.columns.drop(target_variable))

    # 4. Analyze Results (Identify links pointing TO the target variable)
    target_index = var_names.index(target_variable)
    causal_drivers = set()
    p_matrix = results['p_matrix']

    for var_index in range(len(var_names)):
        if var_index == target_index:
            continue
        
        # Check for significant links across lags 1 to tau_max
        # p_matrix dimensions: [driver, target, lag]
        p_values_at_lags = p_matrix[var_index, target_index, 1:]
        
        if np.any(p_values_at_lags <= pc_alpha):
            driver_name = var_names[var_index]
            causal_drivers.add(driver_name)
            print(f"  -> Causal Driver Found: {driver_name} (Min p-value: {np.min(p_values_at_lags):.4f})")

    if not causal_drivers:
        print("WARNING: No significant causal drivers found. Falling back to all features.")
        return list(df.columns.drop(target_variable))

    print(f"Causal Discovery Complete. Drivers: {causal_drivers}")
    return list(causal_drivers)