import numpy as np
import polars as pl
from deap import gp
import logging
import onnxruntime as ort

# Import the GP execution helper from the main framework
try:
    from gp_framework import execute_gp_strategy
    from maml_adapter import CONTEXT_SIZE, FEATURE_SIZE, GP_OUTPUT_SIZE
except ImportError:
    # This allows the script to be run standalone for testing if needed
    def execute_gp_strategy(individual, data_pl, feature_names, pset):
        logging.warning("Running with dummy 'execute_gp_strategy'.")
        return pl.DataFrame({
            'gp_signal': np.random.randn(len(data_pl)),
            'gp_size': np.random.uniform(0.01, 0.1, len(data_pl))
        })
    CONTEXT_SIZE, FEATURE_SIZE, GP_OUTPUT_SIZE = 10, 5, 2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ValidationGauntlet")

class ValidationGauntlet:
    def __init__(self, historical_data_pl: pl.DataFrame, feature_names: list[str], pset: gp.PrimitiveSet, gp_outputs: pl.DataFrame, maml_path: str = None):
        if 'target_return' not in historical_data_pl.columns or 'close' not in historical_data_pl.columns:
            raise ValueError("'target_return' and 'close' must be in the historical data.")
        
        self.historical_data = historical_data_pl
        self.feature_names = feature_names
        self.pset = pset
        self.gp_outputs = gp_outputs
        self.maml_session = self._load_maml_session(maml_path)

    def _load_maml_session(self, path):
        if path:
            try:
                logger.info(f"Loading MAML model from {path}...")
                return ort.InferenceSession(path)
            except Exception as e:
                logger.error(f"Failed to load MAML ONNX model: {e}")
        return None

    def _simulate_hybrid_performance(self, data_slice: pl.DataFrame, gp_outputs_slice: pl.DataFrame):
        """
        Core function to simulate the performance of the hybrid strategy (GP + MAML).
        Returns a Polars Series of PnL values.
        """
        features_np = data_slice.select(self.feature_names).to_numpy().astype(np.float32)
        gp_signal_np = gp_outputs_slice['gp_signal'].to_numpy().astype(np.float32)
        gp_size_np = gp_outputs_slice['gp_size'].to_numpy().astype(np.float32)
        
        signal_mod = np.zeros_like(gp_signal_np)
        size_mod = np.ones_like(gp_size_np)

        if self.maml_session:
            for i in range(CONTEXT_SIZE, len(features_np)):
                context = features_np[i-CONTEXT_SIZE:i].reshape(1, CONTEXT_SIZE, FEATURE_SIZE)
                ort_inputs = {self.maml_session.get_inputs()[0].name: context}
                ort_outs = self.maml_session.run(None, ort_inputs)
                signal_mod[i] = ort_outs[0][0, 0]
                size_mod[i] = ort_outs[1][0, 0]

        # Apply modulation
        final_signal = gp_signal_np + signal_mod
        final_size = gp_size_np * size_mod
        
        # Determine trade action (-1 for Sell, 1 for Buy, 0 for Hold)
        trade_action = np.sign(np.tanh(final_signal)) # Using sign for clear action
        
        # Calculate PnL
        forward_returns = data_slice['target_return'].to_numpy()
        pnl = trade_action * final_size * forward_returns
        
        return pl.Series("pnl", pnl)

    def run(self, individual):
        """Runs all validation tests and returns a final pass/fail decision."""
        logger.info("\n--- Running Advanced Validation Gauntlet ---")
        
        wf_sharpe, is_wf_valid = self.run_walk_forward_validation(individual)
        logger.info(f"[Walk-Forward] Avg Sharpe: {wf_sharpe:.4f}. Threshold > 0.5. Passed: {is_wf_valid}")

        mc_pnl_percentile, is_mc_valid = self.run_monte_carlo_simulation(individual)
        logger.info(f"[Monte Carlo] 5th Percentile PnL: {mc_pnl_percentile:.4f}. Threshold > 0. Passed: {is_mc_valid}")
        
        is_causal_valid = self.run_causal_refutation()
        logger.info(f"[Causal Refutation] Passed: {is_causal_valid}")

        final_decision = is_wf_valid and is_mc_valid and is_causal_valid
        logger.info(f"--- Gauntlet Complete. Final Decision: {'PASSED' if final_decision else 'FAILED'} ---")
        return final_decision

    def _calculate_sharpe_ratio(self, returns: pl.Series):
        """Helper to calculate annualized Sharpe ratio from a Polars Series."""
        if returns.std() == 0 or returns.std() is None:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def run_walk_forward_validation(self, individual, n_splits=5):
        """Performs walk-forward validation."""
        fold_size = len(self.historical_data) // n_splits
        sharpe_ratios = []

        for i in range(1, n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            
            test_data = self.historical_data.slice(start_idx, fold_size)
            test_gp_outputs = self.gp_outputs.slice(start_idx, fold_size)

            if len(test_data) > CONTEXT_SIZE:
                pnl = self._simulate_hybrid_performance(test_data, test_gp_outputs)
                sharpe = self._calculate_sharpe_ratio(pnl)
                sharpe_ratios.append(sharpe)

        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
        return avg_sharpe, avg_sharpe > 0.5

    def run_monte_carlo_simulation(self, individual, n_simulations=200):
        """Performs Monte Carlo simulation on bootstrapped returns."""
        # Simulate once to get the consistent trading actions
        pnl_series = self._simulate_hybrid_performance(self.historical_data, self.gp_outputs)
        
        # We need the effective signal * size to multiply against random returns
        # PnL = (Action * Size) * Return => (Action * Size) = PnL / Return
        # Avoid division by zero
        safe_returns = np.where(self.historical_data['target_return'].to_numpy() == 0, 1e-9, self.historical_data['target_return'].to_numpy())
        trade_leverage = pnl_series.to_numpy() / safe_returns

        historical_returns = self.historical_data['target_return'].to_numpy()
        total_pnls = []

        for _ in range(n_simulations):
            simulated_returns = np.random.choice(historical_returns, size=len(trade_leverage), replace=True)
            total_pnl = np.sum(trade_leverage * simulated_returns)
            total_pnls.append(total_pnl)
            
        pnl_5th_percentile = np.percentile(total_pnls, 5) if total_pnls else 0.0
        return pnl_5th_percentile, pnl_5th_percentile > 0

    def run_causal_refutation(self):
        """
        Uses DoWhy to add a random common cause and check if the causal estimate changes significantly.
        This is a basic test of robustness.
        """
        # Convert to pandas for DoWhy compatibility
        data_pd = self.historical_data.to_pandas()
        
        # Define a simplified causal graph based on the available features
        # We assume all features are potential causes of the target return.
        graph_dot = "digraph { "
        for feature in self.feature_names:
            graph_dot += f'"{feature}" -> "target_return"; '
        graph_dot += "}"

        # Choose a primary feature as the 'treatment' for the test
        treatment_feature = 'imbalance' if 'imbalance' in self.feature_names else self.feature_names[0]

        try:
            model = CausalModel(
                data=data_pd,
                treatment=treatment_feature,
                outcome='target_return',
                graph=graph_dot
            )
            
            # 1. Identify the causal effect
            identified_estimand = model.identify_effect()
            
            # 2. Estimate the effect
            estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
            
            # 3. Refute the estimate by adding an unobserved common cause
            refutation = model.refute_estimate(identified_estimand, estimate, method_name="add_unobserved_common_cause")
            
            # A robust model should not be highly sensitive to this refutation.
            # We check if the p-value is high (i.e., we can't reject the null that the estimate is stable).
            # A p-value > 0.05 suggests the estimate is robust.
            logger.info(f"[Causal Refutation] Refutation p-value: {refutation.p_value:.4f}")
            return refutation.p_value > 0.05

        except Exception as e:
            logger.error(f"[Causal Refutation] Failed due to an error: {e}")
            # Fail validation if the causal analysis itself fails
            return False
