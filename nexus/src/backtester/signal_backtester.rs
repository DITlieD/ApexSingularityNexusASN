use polars::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use anyhow::Result;
use rust_decimal::prelude::ToPrimitive;

// Re-use the configuration and result structures from the vectorized backtester
use super::vectorized_backtester::{BacktestConfig, BacktestResult};

/// Runs a high-speed backtest using pre-computed win probabilities from a Polars DataFrame.
pub fn run_signal_backtest(
    data: &DataFrame,
    config: &BacktestConfig,
) -> Result<BacktestResult> {
    
    let target_capital = config.initial_capital * config.target_capital_factor;
    let num_rows = data.height();
    if num_rows == 0 {
        return Ok(BacktestResult { ttt_fitness: 0.0, final_equity: config.initial_capital });
    }

    // 1. Data Preparation (Expects Decimal types)
    let close_prices = data.column("close")?.decimal()?.clone();
    let win_probs = data.column("win_probability")?.decimal()?.clone();

    // 2. Initialization
    let mut cash = config.initial_capital;
    let mut inventory = dec!(0);
    let mut time_to_target: Option<usize> = None;

    // 3. The Backtest Loop
    for i in 0..num_rows {
        let current_price = match close_prices.get(i) {
            Some(val) => Decimal::from_i128_with_scale(val, close_prices.scale() as u32),
            None => dec!(0),
        };
        if current_price <= dec!(0) { continue; }

        let p = match win_probs.get(i) {
            Some(val) => Decimal::from_i128_with_scale(val, win_probs.scale() as u32),
            None => dec!(0.5),
        };
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

        // B. Update Equity and Check TTT
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

    // 4. Calculate TTT Fitness
    let last_price = match close_prices.get(num_rows - 1) {
        Some(val) => Decimal::from_i128_with_scale(val, close_prices.scale() as u32),
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