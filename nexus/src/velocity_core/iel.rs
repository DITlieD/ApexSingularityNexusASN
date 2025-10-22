use anyhow::Result;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};
use onnxruntime::{session::Session, GraphOptimizationLevel, environment::Environment};
use ndarray::Array;
use std::sync::Arc;

lazy_static::lazy_static! {
    static ref IEL_ONNX_ENV: Arc<Environment> = Arc::new(Environment::builder()
        .with_name("ASN_IEL")
        .build()
        .expect("Failed to initialize IEL ONNX Runtime environment"));
}

// Define the possible execution actions the IEL can decide on
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionAction {
    MarketOrder { side: String, qty: Decimal },
    LimitOrder { side: String, qty: Decimal, price: Decimal },
    Hold,
}

// The Intelligent Executor struct
pub struct IntelligentExecutor {
    model: Session<'static>,
}

impl IntelligentExecutor {
    // Load the ONNX model from the specified path
    pub fn run_iel_model(state: &[Decimal]) -> Result<ExecutionAction, anyhow::Error> {
    let iel_model_path = "iel_model.onnx";
    let model = Session::builder(IEL_ONNX_ENV.clone())?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(iel_model_path)?;
    let input_tensor = Array::from_shape_vec((1, 4), state.to_vec()).unwrap();
    let result = model.run(vec![input_tensor.into()]).unwrap();
    let output_tensor = result[0].try_extract_tensor::<f32>()?;
    let strategy_signal = Decimal::from_f32(output_tensor[[0, 0]]).unwrap_or_default();
    let strategy_size = Decimal::from_f32(output_tensor[[0, 1]]).unwrap_or_default();
    let tactic = output_tensor[[0, 2]] as u32;
    let price = Decimal::from_f32(output_tensor[[0, 3]]).unwrap_or_default();
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
