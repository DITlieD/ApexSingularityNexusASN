use crate::inference::gp_interpreter::{GPStrategy};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use anyhow::{Result, anyhow};
use rust_decimal::prelude::ToPrimitive;
use arrow2::array::{Array, PrimitiveArray};
use arrow2::chunk::Chunk;
use arrow2::datatypes::DataType;

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: Decimal,
    pub target_capital_factor: Decimal,
    pub fee_rate: Decimal,
    pub slippage_factor: Decimal,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: dec!(200.0),
            target_capital_factor: dec!(2.0),
            fee_rate: dec!(0.00055),
            slippage_factor: dec!(0.0001),
        }
    }
}

#[derive(Debug)]
pub struct BacktestResult {
    pub ttt_fitness: f64,
    pub final_equity: Decimal,
}

pub fn run_vectorized_backtest(
    strategy: &GPStrategy,
    data: &Chunk<Box<dyn Array>>,
    schema: &arrow2::datatypes::Schema,
    config: &BacktestConfig,
    feature_names: &[String],
) -> Result<BacktestResult> {
    
    let target_capital = config.initial_capital * config.target_capital_factor;
    let num_rows = data.len();
    if num_rows == 0 {
        return Ok(BacktestResult { ttt_fitness: 0.0, final_equity: config.initial_capital });
    }

    // Find column indices
    let close_idx = schema.fields.iter().position(|f| f.name == "close").ok_or_else(|| anyhow!("'close' column not found"))?;
    let feature_indices: Vec<usize> = feature_names.iter()
        .map(|name| schema.fields.iter().position(|f| f.name == *name).ok_or_else(|| anyhow!(format!("Feature '{}' not found", name))))
        .collect::<Result<_, _>>()?;

    // Extract close prices array
    let close_prices_arr = data.arrays()[close_idx]
        .as_any()
        .downcast_ref::<PrimitiveArray<i128>>()
        .ok_or_else(|| anyhow!("'close' column is not of type Decimal128"))?;
    
    let scale = if let DataType::Decimal(_, scale) = close_prices_arr.data_type() {
        scale
    } else {
        return Err(anyhow!("'close' column is not a Decimal type"));
    };

    // Extract feature data into a row-oriented structure
    let mut feature_data: Vec<Vec<Decimal>> = vec![vec![dec!(0); feature_names.len()]; num_rows];
    for (col_idx, &arrow_idx) in feature_indices.iter().enumerate() {
        let array = data.arrays()[arrow_idx]
            .as_any()
            .downcast_ref::<PrimitiveArray<i128>>()
            .ok_or_else(|| anyhow!(format!("Feature column '{}' is not of type Decimal128", feature_names[col_idx])))?;
        
        let feature_scale = if let DataType::Decimal(_, scale) = array.data_type() {
            scale
        } else {
            return Err(anyhow!(format!("Feature column '{}' is not a Decimal type", feature_names[col_idx])));
        };

        for (row_idx, value) in array.iter().enumerate() {
            if let Some(val) = value {
                feature_data[row_idx][col_idx] = Decimal::from_i128_with_scale(*val, feature_scale as u32);
            }
        }
    }

    let mut cash = config.initial_capital;
    let mut inventory = dec!(0);
    let mut time_to_target: Option<usize> = None;

    for i in 0..num_rows {
        let current_price = match close_prices_arr.get(i) {
            Some(val) => Decimal::from_i128_with_scale(val, scale as u32),
            None => dec!(0),
        };
        if current_price <= dec!(0) { continue; }

        let output = match strategy.evaluate(&feature_data[i]) {
            Ok(o) => o,
            Err(_) => continue,
        };

        let p = output.win_probability;
        let q = dec!(1.0) - p;
        let b = dec!(3.0);
        let optimal_f = (b * p - q) / b;

        if optimal_f > dec!(0.0) {
            let trade_fraction = dec!(0.5) * optimal_f;
            let trade_size_dollars = cash * trade_fraction;
            let qty = trade_size_dollars / current_price;

            if p > dec!(0.5) { // BUY
                let execution_price = current_price * (dec!(1) + config.slippage_factor);
                let cost = qty * execution_price;
                let fee = cost * config.fee_rate;
                if cash >= cost + fee {
                    cash -= cost + fee;
                    inventory += qty;
                }
            } else { // SELL
                let execution_price = current_price * (dec!(1) - config.slippage_factor);
                let size_to_sell = qty.min(inventory);
                if size_to_sell > dec!(0) {
                    let revenue = size_to_sell * execution_price;
                    let fee = revenue * config.fee_rate;
                    cash += revenue - fee;
                    inventory -= size_to_sell;
                }
            }
        }

        let equity = cash + inventory * current_price;
        if equity >= target_capital {
            time_to_target = Some(i + 1);
            break;
        }
        if equity <= dec!(0) {
            time_to_target = Some(num_rows);
            break;
        }
    }

    let last_price = match close_prices_arr.get(num_rows - 1) {
        Some(val) => Decimal::from_i128_with_scale(val, scale as u32),
        None => dec!(0),
    };
    let final_equity = cash + inventory * last_price;

    let ttt_fitness = if let Some(ttt) = time_to_target {
        if final_equity >= target_capital { -(ttt as f64) } 
        else { -(num_rows as f64) * 2.0 }
    } else {
        let equity_ratio = (final_equity / config.initial_capital).to_f64().unwrap_or(0.0);
        -(num_rows as f64) + equity_ratio
    };

    Ok(BacktestResult {
        final_equity,
        ttt_fitness,
    })
}