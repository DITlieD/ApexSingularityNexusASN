use pyo3::prelude::*;
use rust_decimal::prelude::{Decimal, ToPrimitive};
use rust_decimal_macros::dec;
use polars::prelude::DataFrame;
use pyo3_polars::PyDataFrame;

pub mod hf_abm;
pub mod velocity_core;
pub mod inference;
pub mod backtester;

use hf_abm::simulator::{SimulationConfig, MarketStatistics, run_accelerated_simulation, run_accelerated_chimera_simulation};
use backtester::signal_backtester::run_signal_backtest;
use backtester::vectorized_backtester::{BacktestConfig, run_vectorized_backtest};
use inference::gp_interpreter::{GPStrategy, HybridStrategy};

#[pyclass(name = "SimulationConfig")]
#[derive(Clone)]
struct PySimulationConfig {
    #[pyo3(get, set)]
    num_ticks: u64,
    #[pyo3(get, set)]
    taker_fee_rate: f64,
    #[pyo3(get, set)]
    maker_fee_rate: f64,
    #[pyo3(get, set)]
    initial_capital: f64,
    #[pyo3(get, set)]
    target_capital_factor: f64,
}

#[pymethods]
impl PySimulationConfig {
    #[new]
    fn new(num_ticks: u64, taker_fee_rate: f64, maker_fee_rate: f64, initial_capital: f64, target_capital_factor: f64) -> Self {
        Self { num_ticks, taker_fee_rate, maker_fee_rate, initial_capital, target_capital_factor }
    }
}

impl From<PySimulationConfig> for SimulationConfig {
    fn from(py_config: PySimulationConfig) -> Self {
        SimulationConfig {
            num_ticks: py_config.num_ticks,
            taker_fee_rate: Decimal::from_f64_retain(py_config.taker_fee_rate).unwrap_or(dec!(0.00055)),
            maker_fee_rate: Decimal::from_f64_retain(py_config.maker_fee_rate).unwrap_or(dec!(0.0002)),
            initial_capital: Decimal::from_f64_retain(py_config.initial_capital).unwrap_or(dec!(200.0)),
            target_capital_factor: Decimal::from_f64_retain(py_config.target_capital_factor).unwrap_or(dec!(2.0)),
            slippage_factor: dec!(0.0001), // Default slippage
        }
    }
}

#[pyclass(name = "BacktestConfig")]
#[derive(Clone)]
struct PyBacktestConfig {
    #[pyo3(get, set)]
    initial_capital: f64,
    #[pyo3(get, set)]
    target_capital_factor: f64,
    #[pyo3(get, set)]
    fee_rate: f64,
    #[pyo3(get, set)]
    slippage_factor: f64,
}

#[pymethods]
impl PyBacktestConfig {
    #[new]
    fn new(initial_capital: f64, target_capital_factor: f64, fee_rate: f64, slippage_factor: f64) -> Self {
        Self { initial_capital, target_capital_factor, fee_rate, slippage_factor }
    }
}

impl From<PyBacktestConfig> for BacktestConfig {
    fn from(py_config: PyBacktestConfig) -> Self {
        BacktestConfig {
            initial_capital: Decimal::from_f64_retain(py_config.initial_capital).unwrap_or(dec!(200.0)),
            target_capital_factor: Decimal::from_f64_retain(py_config.target_capital_factor).unwrap_or(dec!(2.0)),
            fee_rate: Decimal::from_f64_retain(py_config.fee_rate).unwrap_or(dec!(0.00055)),
            slippage_factor: Decimal::from_f64_retain(py_config.slippage_factor).unwrap_or(dec!(0.0001)),
        }
    }
}

#[pyclass(name = "MarketStatistics")]
struct PyMarketStatistics {
    #[pyo3(get)]
    total_volume: f64,
    #[pyo3(get)]
    realized_volatility: f64,
    #[pyo3(get)]
    avg_spread: f64,
    #[pyo3(get)]
    trade_count: u64,
}

impl From<MarketStatistics> for PyMarketStatistics {
    fn from(stats: MarketStatistics) -> Self {
        Self {
            total_volume: stats.total_volume.to_f64().unwrap_or(0.0),
            realized_volatility: stats.realized_volatility.to_f64().unwrap_or(0.0),
            avg_spread: stats.avg_spread.to_f64().unwrap_or(0.0),
            trade_count: stats.trade_count,
        }
    }
}

#[pyfunction]
fn run_vectorized_backtest_py(strategy_json: String, data: PyDataFrame, config: PyBacktestConfig, feature_names: Vec<String>) -> PyResult<f64> {
    let strategy = GPStrategy::from_json(&strategy_json).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid strategy JSON: {}", e)))?;
    let rust_config: BacktestConfig = config.into();
    let rust_df: DataFrame = data.into();
    let result = Python::with_gil(|py| py.allow_threads(|| run_vectorized_backtest(&strategy, &rust_df, &rust_config, &feature_names)));
    let backtest_result = result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Backtest failed: {}", e)))?;
    Ok(backtest_result.ttt_fitness)
}

#[pyfunction]
fn run_signal_backtest_py(data: PyDataFrame, config: PyBacktestConfig) -> PyResult<f64> {
    let rust_config: BacktestConfig = config.into();
    let rust_df: DataFrame = data.into();
    let result = Python::with_gil(|py| py.allow_threads(|| run_signal_backtest(&rust_df, &rust_config)));
    let backtest_result = result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Signal Backtest failed: {}", e)))?;
    Ok(backtest_result.ttt_fitness)
}

#[pyfunction]
#[pyo3(signature = (config, apex_strategy_json, apex_onnx_path, dsg_parameters))]
fn run_accelerated_simulation_py(config: PySimulationConfig, apex_strategy_json: String, apex_onnx_path: Option<String>, dsg_parameters: Vec<f64>) -> PyResult<f64> {
    let rust_config: SimulationConfig = config.into();
    let result = run_accelerated_simulation(rust_config, apex_strategy_json, apex_onnx_path, dsg_parameters);
    Ok(result.ttt_fitness)
}

#[pyfunction]
fn run_accelerated_chimera_simulation_py(config: PySimulationConfig, dsg_parameters: Vec<f64>) -> PyResult<PyMarketStatistics> {
    let rust_config: SimulationConfig = config.into();
    let stats = run_accelerated_chimera_simulation(rust_config, dsg_parameters);
    Ok(stats.into())
}

#[pymodule]
fn nexus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulationConfig>()?;
    m.add_class::<PyMarketStatistics>()?;
    m.add_function(wrap_pyfunction!(run_accelerated_simulation_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_accelerated_chimera_simulation_py, m)?)?;
    m.add_class::<PyBacktestConfig>()?;
    m.add_function(wrap_pyfunction!(run_vectorized_backtest_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_signal_backtest_py, m)?)?;
    Ok(())
}