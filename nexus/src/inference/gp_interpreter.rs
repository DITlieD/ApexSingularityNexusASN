use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error;
use ort::{value::Value as OrtValue, session::Session, session::builder::GraphOptimizationLevel};
use ndarray::Array;
use std::sync::Arc;

// ... (Errors and Primitive enum remain the same as handover) ...

#[derive(Error, Debug)]
pub enum InterpreterError {
    #[error("Failed to deserialize strategy: {0}")]
    Deserialization(#[from] serde_json::Error),
    #[error("Invalid node structure found")]
    InvalidNode,
    #[error("Evaluation failed: {0}")]
    Evaluation(String),
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Primitive {
    // ... (Primitives remain the same) ...
    #[serde(rename = "add")]
    Add,
    #[serde(rename = "sub")]
    Sub,
    #[serde(rename = "mul")]
    Mul,
    #[serde(rename = "protectedDiv")]
    ProtectedDiv,
    #[serde(rename = "ifThenElse")]
    IfThenElse,
    #[serde(rename = "neg")]
    Neg,
    Feature(usize),
    Constant(f64),
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Node {
    primitive: Primitive,
    children: Vec<Node>,
}

// NEW: Struct for the evaluation result
#[derive(Debug, Clone, Copy)]
pub struct StrategyOutput {
    pub win_probability: Decimal,
}

// MODIFIED: Contains a single root for win probability
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GPStrategy {
    win_prob_root: Node,
}

use rust_decimal::prelude::ToPrimitive;

pub struct HybridStrategy {
    gp_strategy: GPStrategy,
    onnx_path: Option<String>,
}

impl HybridStrategy {
    pub fn new(gp_json_str: &str, onnx_path: Option<String>) -> Result<Self, InterpreterError> {
        let gp_strategy = GPStrategy::from_json(gp_json_str)?;
        Ok(Self { gp_strategy, onnx_path })
    }

    pub fn evaluate(&self, features: &[Decimal], external_features: Option<&[f32]>) -> Result<StrategyOutput, InterpreterError> {
        let gp_output = self.gp_strategy.evaluate(features)?;
        if let (Some(path), Some(ext_feats)) = (&self.onnx_path, external_features) {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(path)?;

            let input_tensor = Array::from_shape_vec((1, ext_feats.len()), ext_feats.to_vec())
                .map_err(|e| InterpreterError::Evaluation(e.to_string()))?;
            
            let inputs = [OrtValue::from_array(input_tensor.into_dyn()).unwrap()];
            let results = session.run(&inputs)?;
            
            if let Ok(output_tensor) = results[0].try_extract_tensor::<f32>() {
                let view = output_tensor.view();
                let onnx_logit = view.iter().next().unwrap_or(&0.0);
                let combined_logit = gp_output.win_probability.to_f64().unwrap_or(0.5) + *onnx_logit as f64;
                let probability = Decimal::from_f64_retain(1.0 / (1.0 + (-combined_logit).exp())).unwrap_or(dec!(0.5));
                Ok(StrategyOutput { win_probability: probability })
            } else {
                Ok(gp_output)
            }
        } else {
            Ok(gp_output)
        }
    }
}

impl GPStrategy {
    pub fn from_json(json_str: &str) -> Result<Self, InterpreterError> {
        let strategy: GPStrategy = serde_json::from_str(json_str)?;
        Ok(strategy)
    }

    /// Evaluates the GP tree and clamps the output to a valid probability range.
    pub fn evaluate(&self, features: &[Decimal]) -> Result<StrategyOutput, InterpreterError> {
        let raw_output = Self::evaluate_node(&self.win_prob_root, features)?;

        // Sigmoid activation for Decimal: 1 / (1 + exp(-x))
        let probability = if let Some(raw_f64) = raw_output.to_f64() {
            let exp_val = (-raw_f64).exp();
            Decimal::from_f64_retain(1.0 / (1.0 + exp_val)).unwrap_or(dec!(0.5))
        } else {
            dec!(0.5) // Fallback for extreme values
        };

        Ok(StrategyOutput {
            win_probability: probability,
        })
    }

    // The recursive evaluation logic remains the same as the handover file, with minor safety checks.
    fn evaluate_node(node: &Node, features: &[Decimal]) -> Result<Decimal, InterpreterError> {
        match &node.primitive {
            Primitive::Add => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                Ok(left.checked_add(right).unwrap_or(Decimal::ZERO))
            }
            Primitive::Sub => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                Ok(left.checked_sub(right).unwrap_or(Decimal::ZERO))
            }
            Primitive::Mul => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                Ok(left.checked_mul(right).unwrap_or(Decimal::ZERO))
            }
            Primitive::ProtectedDiv => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                if right.is_zero() {
                    Ok(Decimal::ONE)
                } else {
                    Ok(left.checked_div(right).unwrap_or(Decimal::ONE))
                }
            }
            Primitive::IfThenElse => {
                let condition = Self::evaluate_node(&node.children[0], features)?;
                // If condition > 0 (matching Python interpretation), execute the first branch.
                if condition > Decimal::ZERO {
                    Self::evaluate_node(&node.children[1], features)
                } else {
                    Self::evaluate_node(&node.children[2], features)
                }
            }
            Primitive::Neg => {
                let child = Self::evaluate_node(&node.children[0], features)?;
                Ok(-child)
            }
            Primitive::Feature(index) => {
                features.get(*index).cloned().ok_or_else(|| {
                    InterpreterError::Evaluation(format!("Feature index {} out of bounds", index))
                })
            }
            Primitive::Constant(value) => {
                Decimal::from_f64_retain(*value).ok_or_else(|| {
                    InterpreterError::Evaluation(format!("Invalid constant value: {}", value))
                })
            }
        }
    }
}