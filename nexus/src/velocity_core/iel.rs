use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use ort::{value::Value as OrtValue, session::Session, session::builder::GraphOptimizationLevel};
use ndarray::Array;
use std::sync::Arc;

// Define the possible execution actions the IEL can decide on
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionAction {
    MarketOrder { side: String, qty: Decimal },
    LimitOrder { side: String, qty: Decimal, price: Decimal },
    Hold,
}

// The Intelligent Executor struct
pub struct IntelligentExecutor {
    model: Session,
}

impl IntelligentExecutor {
    // Load the ONNX model from the specified path
    pub fn run_iel_model(state: &[Decimal]) -> Result<ExecutionAction, anyhow::Error> {
        let iel_model_path = "iel_model.onnx";
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(iel_model_path)?;
        
        let state_f32: Vec<f32> = state.iter().map(|d| d.to_f32().unwrap_or(0.0)).collect();
        let input_tensor = Array::from_shape_vec((1, 4), state_f32).unwrap();
        let inputs = [OrtValue::from_array(input_tensor.into_dyn()).unwrap()];
        let result = model.run(&inputs)?;
        let output_tensor = result[0].try_extract_tensor::<f32>()?;
        let view = output_tensor.view();
        let strategy_signal = Decimal::from_f32(view[[0, 0]]).unwrap_or_default();
        let strategy_size = Decimal::from_f32(view[[0, 1]]).unwrap_or_default();
        let tactic = view[[0, 2]] as u32;
        let price = Decimal::from_f32(view[[0, 3]]).unwrap_or_default();
        let qty_multiplier = 1.0;

        let side = if strategy_signal > Decimal::ZERO { "Buy" } else { "Sell" }.to_string();
        let qty = strategy_size * Decimal::from_f64(qty_multiplier).unwrap_or(dec!(1.0));

        match tactic {
            0 => Ok(ExecutionAction::MarketOrder { side, qty }),
            1 => Ok(ExecutionAction::LimitOrder { side, qty, price }),
            _ => Ok(ExecutionAction::Hold),
        }
    }
}
