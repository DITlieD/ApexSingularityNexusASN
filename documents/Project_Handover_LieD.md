# Project Handover: The Apex Singularity Nexus (ASN)

**Date:** 2025-10-22

This document provides a complete snapshot of the ASN project for handover. It includes the full source code for all components and the detailed roadmap for achieving 100% completion as defined by the project blueprint.

---

## I. Current Project State: Source Code

This section contains the complete source code for the two main components of the project:
1.  **Nexus:** The core trading engine and simulator, written in Rust.
2.  **Forge:** The research and strategy evolution environment, written in Python.

### 1. The `nexus` Crate (Rust Core)

#### **File: `nexus/Cargo.toml`**
```toml
[package]
name = "nexus"
version = "0.1.0"
edition = "2021"

[lib]
name = "nexus"
crate-type = ["cdylib", "rlib"]

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[[bin]]
name = "nexus_tester"
path = "src/bin/main.rs"

[dependencies]
tokio = { version = "1", features = ["full"] }
tokio-tungstenite = { version = "0.23.1", features = ["native-tls"] }
futures-util = "0.3.30"
url = "2.5.2"

polars = { version = "0.41.0", features = ["lazy", "ndarray", "serde"] }

serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_qs = "0.12"

rust_decimal = "1.35"
rust_decimal_macros = "1.34"

reqwest = { version = "0.12", features = ["json"] }
hmac = "0.12"
sha2 = "0.10"
hex = "0.4"
uuid = { version = "1.8", features = ["v4"] }

pyo3 = { version = "0.21.2", features = ["extension-module"] }

anyhow = "1.0"
thiserror = "1.0"

```

#### **File: `nexus/src/lib.rs`**
```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rust_decimal::prelude::{Decimal, ToPrimitive};
use std::collections::BTreeMap;

pub mod hf_abm;
pub mod velocity_core;
pub mod inference;

use hf_abm::simulator::{self as hfabm, Agent, MarketState, Order, Simulation, Trade};

// --- Python Wrapper for Agent Trait ---

/// This struct holds a Python object that is expected to have an `on_tick` method.
/// It implements the Rust `Agent` trait, acting as a bridge.
struct PyAgent {
    id: u64,
    py_agent: PyObject,
}

impl Agent for PyAgent {
    fn id(&self) -> u64 {
        self.id
    }

    /// This is the core of the bridge: it calls the Python agent's `on_tick` method from Rust.
    fn on_tick(&mut self, market_state: &MarketState) -> Vec<Order> {
        Python::with_gil(|py| {
            let market_state_py = PyMarketState::from(market_state);
            let market_state_py_obj = Py::new(py, market_state_py).unwrap().to_object(py);

            match self.py_agent.call_method1(py, "on_tick", (market_state_py_obj,)) {
                Ok(py_orders) => {
                    let bound_orders = py_orders.bind(py);
                    // Convert the list of Python Order objects back to Rust Order structs
                    if let Ok(list) = bound_orders.downcast::<PyList>() {
                        list.iter()
                            .filter_map(|py_obj| {
                                if let Ok(order_py) = py_obj.extract::<PyRef<PyOrder>>() {
                                    Some(order_py.inner)
                                } else {
                                    None
                                }
                            })
                            .collect()
                    } else {
                        vec![]
                    }
                }
                Err(e) => {
                    e.print_and_set_sys_last_vars(py);
                    vec![]
                }
            }
        })
    }
}

// --- PyO3 Exposed Classes ---

#[pyclass(name = "Order")]
#[derive(Clone, Copy)]
struct PyOrder {
    #[pyo3(get, set)]
    agent_id: u64,
    #[pyo3(get, set)]
    price: f64,
    #[pyo3(get, set)]
    size: f64,
    #[pyo3(get, set)]
    is_bid: bool,
    // Keep the inner Rust struct for easy conversion
    inner: Order,
}

#[pymethods]
impl PyOrder {
    #[new]
    fn new(agent_id: u64, price: f64, size: f64, is_bid: bool) -> Self {
        let inner = Order {
            agent_id,
            price: Decimal::from_f64_retain(price).unwrap(),
            size: Decimal::from_f64_retain(size).unwrap(),
            is_bid,
        };
        Self { agent_id, price, size, is_bid, inner }
    }
}

#[pyclass(name = "Trade")]
#[derive(Clone, Copy)]
struct PyTrade {
    #[pyo3(get)]
    aggressor_agent_id: u64,
    #[pyo3(get)]
    resting_agent_id: u64,
    #[pyo3(get)]
    price: f64,
    #[pyo3(get)]
    size: f64,
}

impl From<Trade> for PyTrade {
    fn from(trade: Trade) -> Self {
        Self {
            aggressor_agent_id: trade.aggressor_agent_id,
            resting_agent_id: trade.resting_agent_id,
            price: trade.price.to_f64().unwrap(),
            size: trade.size.to_f64().unwrap(),
        }
    }
}

#[pyclass(name = "MarketState")]
struct PyMarketState {
    #[pyo3(get)]
    bids: PyObject,
    #[pyo3(get)]
    asks: PyObject,
    #[pyo3(get)]
    sequence: u64,
}

impl From<&MarketState> for PyMarketState {
    fn from(market_state: &MarketState) -> Self {
        Python::with_gil(|py| {
            let bids_dict = PyDict::new_bound(py);
            for (price, orders) in market_state.bids.iter() {
                let price_f64 = price.0.to_f64().unwrap();
                let size_sum: f64 = orders.iter().map(|o| o.size.to_f64().unwrap()).sum();
                bids_dict.set_item(price_f64, size_sum).unwrap();
            }

            let asks_dict = PyDict::new_bound(py);
            for (price, orders) in market_state.asks.iter() {
                let price_f64 = price.to_f64().unwrap();
                let size_sum: f64 = orders.iter().map(|o| o.size.to_f64().unwrap()).sum();
                asks_dict.set_item(price_f64, size_sum).unwrap();
            }

            Self {
                bids: bids_dict.to_object(py),
                asks: asks_dict.to_object(py),
                sequence: market_state.sequence,
            }
        })
    }
}

#[pyclass(name = "Simulation")]
struct PySimulation {
    sim: Simulation,
}

#[pymethods]
impl PySimulation {
    #[new]
    fn new() -> Self {
        Self {
            sim: Simulation::new(),
        }
    }

    fn add_agent(&mut self, agent: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            let id: u64 = agent.getattr(py, "id")?.extract(py)?;
            let py_agent = PyAgent { id, py_agent: agent };
            self.sim.add_agent(Box::new(py_agent));
            Ok(())
        })
    }

    fn tick(&mut self) -> Vec<PyTrade> {
        self.sim.tick().into_iter().map(PyTrade::from).collect()
    }

    fn get_market_state(&self) -> PyMarketState {
        PyMarketState::from(&self.sim.market_state)
    }
}

/// The main Python module definition.
#[pymodule]
fn nexus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulation>()?;
    m.add_class::<PyOrder>()?;
    m.add_class::<PyTrade>()?;
    m.add_class::<PyMarketState>()?;
    Ok(())
}

```

#### **File: `nexus/src/bin/main.rs`**
```rust
use anyhow::Result;
use nexus::{
    inference::gp_interpreter::GPStrategy,
    velocity_core::{
        execution::{ExecutionClient},
        l2_handler::{OrderBook},
        models::{BybitResponse, SubscriptionMessage},
    },
};
use rust_decimal::Decimal;
use std::fs;
use tokio_tungstenite::connect_async;
use futures_util::{StreamExt, SinkExt};
use serde_json::json;

const BYBIT_WS_URL: &str = "wss://stream-testnet.bybit.com/v5/public/spot";

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- ASN NEXUS INITIALIZING ---");

    // --- 1. Load API Credentials ---
    // IMPORTANT: Ensure your keys are set correctly here.
    let api_key = "DthZpR71I6y6bVKz72".to_string();
    let api_secret = "ZS6eQ2edGQAyMu28p6QrytOUdxkSLOsvCcsV".to_string();

    if api_key == "YOUR_API_KEY" || api_secret == "YOUR_API_SECRET" {
        println!("FATAL: API keys not set in src/bin/main.rs. Exiting.");
        return Ok(())
    }
    
    // --- 2. Initialize Core Components ---
    let execution_client = ExecutionClient::new(api_key, api_secret);
    println!("[OK] Execution Client initialized.");

    let strategy_json = fs::read_to_string("../forge/strategy.json")?;
    let strategy = GPStrategy::from_json(&strategy_json)?;
    println!("[OK] GP Strategy loaded from strategy.json.");

    // --- 3. Connect to WebSocket and Start Main Loop ---
    println!("Connecting to Bybit Testnet WebSocket...");
    let (ws_stream, _) = connect_async(BYBIT_WS_URL).await?;
    println!("[OK] WebSocket connection established.");

    let (mut write, mut read) = ws_stream.split();

    let symbol = "BTCUSDT".to_string();
    let mut order_book = OrderBook::new(symbol.clone());

    let subscription = SubscriptionMessage {
        op: "subscribe",
        args: vec![format!("orderbook.50.{}", symbol)],
    };
    let sub_msg = serde_json::to_string(&subscription)?;
    write.send(sub_msg.into()).await?;
    println!("[OK] Subscribed to L2 order book for {}.", symbol);
    
    println!("\n--- NEXUS IS LIVE ---");

    while let Some(msg) = read.next().await {
        if let Ok(tokio_tungstenite::tungstenite::Message::Text(text)) = msg {
            if let Ok(BybitResponse::OrderBook(resp)) = serde_json::from_str::<BybitResponse>(&text) {
                
                // a. Update Order Book
                match resp.response_type.as_str() {
                    "snapshot" => order_book.apply_snapshot(&resp.data.bids, &resp.data.asks),
                    "delta" => order_book.apply_delta(&resp.data.bids, &resp.data.asks),
                    _ => continue,
                }

                // b. Calculate Features
                if let Some(features) = order_book.calculate_microstructure_features() {
                    let feature_vec = vec![
                        features.top_of_book_imbalance,
                        features.book_pressure,
                        features.clv_proxy,
                    ];

                    // c. Evaluate Strategy
                    let signal = strategy.evaluate(&feature_vec)?;
                    
                    println!("Features: [Imbalance: {:.4}, Pressure: {:.4}, CLV: {:.4}] -> Signal: {:.4}", 
                        features.top_of_book_imbalance, features.book_pressure, features.clv_proxy, signal);

                    // d. Execute Order
                    let order_payload = if signal > Decimal::new(5, 1) { // Signal > 0.5
                        println!("  -> Decision: BUY");
                        Some(json!({
                            "category": "spot", "symbol": &symbol, "side": "Buy",
                            "orderType": "Market", "qty": "0.010"
                        }))
                    } else if signal < Decimal::new(-5, 1) { // Signal < -0.5
                        println!("  -> Decision: SELL");
                        Some(json!({
                            "category": "spot", "symbol": &symbol, "side": "Sell",
                            "orderType": "Market", "qty": "0.010"
                        }))
                    } else {
                        None
                    };

                    if let Some(payload) = order_payload {
                        println!("   -> PLACING ORDER: {}", payload);
                        match execution_client.place_order(&payload).await {
                            Ok(response) => println!("   -> SUCCESS: Order ID {}", response.result.order_id),
                            Err(e) => println!("   -> ERROR: {}", e),
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

```

#### **File: `nexus/src/hf_abm/mod.rs`**
```rust
pub mod simulator;

```

#### **File: `nexus/src/hf_abm/simulator.rs`**
```rust
use rust_decimal::Decimal;
use std::collections::BTreeMap;

// --- Core Data Structures ---

/// Represents a single limit order in the order book.
#[derive(Debug, Clone, Copy)]
pub struct Order {
    pub agent_id: u64,
    pub price: Decimal,
    pub size: Decimal,
    pub is_bid: bool,
}

/// Represents a single trade event that occurred in the matching engine.
#[derive(Debug, Clone, Copy)]
pub struct Trade {
    pub aggressor_agent_id: u64,
    pub resting_agent_id: u64,
    pub price: Decimal,
    pub size: Decimal,
}

/// Represents the current state of the simulated market.
#[derive(Debug)]
pub struct MarketState {
    /// The L2 order book. Bids are high-to-low, Asks are low-to-high.
    /// The Vec<Order> at each price level represents the queue of orders.
    pub bids: BTreeMap<std::cmp::Reverse<Decimal>, Vec<Order>>,
    pub asks: BTreeMap<Decimal, Vec<Order>>,
    
    /// The sequence number for the next event.
    pub sequence: u64,
}

impl MarketState {
    pub fn new() -> Self {
        MarketState {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            sequence: 0,
        }
    }
}

/// Defines the behavior for an agent participating in the market simulation.
pub trait Agent: Send {
    /// Called on each simulation tick, allowing the agent to react to the market.
    fn on_tick(&mut self, market_state: &MarketState) -> Vec<Order>;
    fn id(&self) -> u64;
}


// --- Matching Engine ---

pub struct MatchingEngine;

impl MatchingEngine {
    /// Processes a new order, matches it against the book, and returns any trades that occurred.
    /// Any remaining size is added to the order book.
    pub fn process_order(market_state: &mut MarketState, mut order: Order) -> Vec<Trade> {
        let mut trades = Vec::new();
        
        if order.is_bid {
            Self::match_bid(market_state, &mut order, &mut trades);
        } else {
            Self::match_ask(market_state, &mut order, &mut trades);
        }

        // If the order is not fully filled, add it to the book.
        if order.size > Decimal::ZERO {
            Self::add_order_to_book(market_state, order);
        }

        trades
    }

    fn match_bid(market_state: &mut MarketState, aggressing_order: &mut Order, trades: &mut Vec<Trade>) {
        let mut filled_levels = Vec::new();
        
        // Iterate over ask levels that could match the bid price
        for (ask_price, orders_at_level) in market_state.asks.iter_mut() {
            if aggressing_order.price < *ask_price {
                break; // No more matches possible
            }

            // Iterate through orders at this price level (FIFO)
            for resting_order in orders_at_level.iter_mut() {
                let trade_size = aggressing_order.size.min(resting_order.size);

                trades.push(Trade {
                    aggressor_agent_id: aggressing_order.agent_id,
                    resting_agent_id: resting_order.agent_id,
                    price: resting_order.price,
                    size: trade_size,
                });

                aggressing_order.size -= trade_size;
                resting_order.size -= trade_size;

                if aggressing_order.size == Decimal::ZERO {
                    break;
                }
            }

            // Remove filled orders from the level
            orders_at_level.retain(|o| o.size > Decimal::ZERO);

            if orders_at_level.is_empty() {
                filled_levels.push(*ask_price);
            }

            if aggressing_order.size == Decimal::ZERO {
                break;
            }
        }

        // Clean up empty price levels
        for price in filled_levels {
            market_state.asks.remove(&price);
        }
    }

    fn match_ask(market_state: &mut MarketState, aggressing_order: &mut Order, trades: &mut Vec<Trade>) {
        let mut filled_levels = Vec::new();

        // Iterate over bid levels that could match the ask price
        for (bid_price, orders_at_level) in market_state.bids.iter_mut() {
            if aggressing_order.price > bid_price.0 {
                break; // No more matches possible
            }

            // Iterate through orders at this price level (FIFO)
            for resting_order in orders_at_level.iter_mut() {
                let trade_size = aggressing_order.size.min(resting_order.size);

                trades.push(Trade {
                    aggressor_agent_id: aggressing_order.agent_id,
                    resting_agent_id: resting_order.agent_id,
                    price: resting_order.price,
                    size: trade_size,
                });

                aggressing_order.size -= trade_size;
                resting_order.size -= trade_size;

                if aggressing_order.size == Decimal::ZERO {
                    break;
                }
            }

            // Remove filled orders from the level
            orders_at_level.retain(|o| o.size > Decimal::ZERO);

            if orders_at_level.is_empty() {
                filled_levels.push(*bid_price);
            }

            if aggressing_order.size == Decimal::ZERO {
                break;
            }
        }

        // Clean up empty price levels
        for price in filled_levels {
            market_state.bids.remove(&price);
        }
    }

    /// Adds a new order to the correct price level in the book.
    fn add_order_to_book(market_state: &mut MarketState, order: Order) {
        if order.is_bid {
            market_state.bids.entry(std::cmp::Reverse(order.price)).or_default().push(order);
        } else {
            market_state.asks.entry(order.price).or_default().push(order);
        }
    }
}


// --- Simulation Framework ---

pub struct Simulation {
    pub market_state: MarketState,
    pub agents: Vec<Box<dyn Agent>>,
}

impl Simulation {
    pub fn new() -> Self {
        Simulation {
            market_state: MarketState::new(),
            agents: Vec::new(),
        }
    }

    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        self.agents.push(agent);
    }

    /// Advances the simulation by one tick.
    /// In each tick, every agent is polled for new orders, which are then processed.
    pub fn tick(&mut self) -> Vec<Trade> {
        let mut all_new_orders = Vec::new();
        for agent in self.agents.iter_mut() {
            let agent_orders = agent.on_tick(&self.market_state);
            all_new_orders.extend(agent_orders);
        }

        let mut all_trades = Vec::new();
        for order in all_new_orders {
            let trades = MatchingEngine::process_order(&mut self.market_state, order);
            all_trades.extend(trades);
        }
        
        self.market_state.sequence += 1;
        all_trades
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    // --- Test Agents ---
    struct StaticAgent {
        id: u64,
        orders_to_submit: Vec<Order>,
    }

    impl Agent for StaticAgent {
        fn on_tick(&mut self, _market_state: &MarketState) -> Vec<Order> {
            self.orders_to_submit.drain(..).collect()
        }
        fn id(&self) -> u64 {
            self.id
        }
    }

    // --- Tests ---

    #[test]
    fn test_matching_engine_simple_fill() {
        let mut market_state = MarketState::new();

        // Add a resting ask order
        let resting_ask = Order { agent_id: 1, price: dec!(100.5), size: dec!(10), is_bid: false };
        MatchingEngine::process_order(&mut market_state, resting_ask);

        assert_eq!(market_state.asks.len(), 1);
        assert_eq!(market_state.bids.len(), 0);

        // Add an aggressive bid order that should partially fill the resting ask
        let aggressive_bid = Order { agent_id: 2, price: dec!(101.0), size: dec!(5), is_bid: true };
        let trades = MatchingEngine::process_order(&mut market_state, aggressive_bid);

        // 1. Check if a trade occurred
        assert_eq!(trades.len(), 1);
        let trade = trades[0];
        assert_eq!(trade.price, dec!(100.5));
        assert_eq!(trade.size, dec!(5));
        assert_eq!(trade.aggressor_agent_id, 2);
        assert_eq!(trade.resting_agent_id, 1);

        // 2. Check if the market state is correct
        assert_eq!(market_state.asks.len(), 1);
        assert_eq!(market_state.bids.len(), 0);
        
        let (price, orders) = market_state.asks.iter().next().unwrap();
        assert_eq!(*price, dec!(100.5));
        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].size, dec!(5)); // 10 (initial) - 5 (filled)
    }

    #[test]
    fn test_matching_engine_full_fill_and_add_to_book() {
        let mut market_state = MarketState::new();

        // Add two resting ask orders
        let resting_ask_1 = Order { agent_id: 1, price: dec!(100.5), size: dec!(10), is_bid: false };
        let resting_ask_2 = Order { agent_id: 2, price: dec!(101.0), size: dec!(10), is_bid: false };
        MatchingEngine::process_order(&mut market_state, resting_ask_1);
        MatchingEngine::process_order(&mut market_state, resting_ask_2);

        assert_eq!(market_state.asks.len(), 2);

        // Add an aggressive bid that will fill the first ask and partially fill the second,
        // then rest on the book.
        let aggressive_bid = Order { agent_id: 3, price: dec!(102.0), size: dec!(25), is_bid: true };
        let trades = MatchingEngine::process_order(&mut market_state, aggressive_bid);

        // 1. Check trades
        assert_eq!(trades.len(), 2);
        // Trade 1 (fills first ask)
        assert_eq!(trades[0].price, dec!(100.5));
        assert_eq!(trades[0].size, dec!(10));
        // Trade 2 (fills second ask)
        assert_eq!(trades[1].price, dec!(101.0));
        assert_eq!(trades[1].size, dec!(10));

        // 2. Check market state
        assert_eq!(market_state.asks.len(), 0); // Both asks should be filled and removed
        assert_eq!(market_state.bids.len(), 1); // The remainder of the aggressive bid should be on the book

        let (price, orders) = market_state.bids.iter().next().unwrap();
        assert_eq!(price.0, dec!(102.0));
        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].size, dec!(5)); // 25 (initial) - 10 (fill 1) - 10 (fill 2)
        assert_eq!(orders[0].agent_id, 3);
    }

    #[test]
    fn test_simulation_tick() {
        let mut simulation = Simulation::new();

        // Agent 1 wants to sell 10 units at 100.5
        let agent1 = Box::new(StaticAgent {
            id: 1,
            orders_to_submit: vec![
                Order { agent_id: 1, price: dec!(100.5), size: dec!(10), is_bid: false }
            ],
        });
        simulation.add_agent(agent1);

        // Agent 2 wants to buy 5 units at 101.0
        let agent2 = Box::new(StaticAgent {
            id: 2,
            orders_to_submit: vec![
                Order { agent_id: 2, price: dec!(101.0), size: dec!(5), is_bid: true }
            ],
        });
        simulation.add_agent(agent2);

        // --- Tick 1 ---
        // Both agents submit their orders. The buy order from agent 2 should match the sell from agent 1.
        let trades = simulation.tick();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, dec!(100.5));
        assert_eq!(trades[0].size, dec!(5));

        // Check the book state
        assert_eq!(simulation.market_state.bids.len(), 0);
        assert_eq!(simulation.market_state.asks.len(), 1);
        let (_, asks) = simulation.market_state.asks.iter().next().unwrap();
        assert_eq!(asks[0].size, dec!(5)); // Partially filled
        assert_eq!(asks[0].agent_id, 1);
        assert_eq!(simulation.market_state.sequence, 1);
    }
}

```

#### **File: `nexus/src/inference/mod.rs`**
```rust
pub mod gp_interpreter;

```

#### **File: `nexus/src/inference/gp_interpreter.rs`**
```rust
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error;

// --- Errors ---
#[derive(Error, Debug)]
pub enum InterpreterError {
    #[error("Failed to deserialize strategy: {0}")]
    Deserialization(#[from] serde_json::Error),
    #[error("Invalid node structure found")]
    InvalidNode,
    #[error("Evaluation failed: {0}")]
    Evaluation(String),
}

// --- GP Tree Representation ---

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Primitive {
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GPStrategy {
    root: Node,
}

impl GPStrategy {
    /// Creates a new strategy from a JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, InterpreterError> {
        let strategy: GPStrategy = serde_json::from_str(json_str)?;
        Ok(strategy)
    }

    /// Evaluates the GP tree with the given input features.
    pub fn evaluate(&self, features: &[Decimal]) -> Result<Decimal, InterpreterError> {
        Self::evaluate_node(&self.root, features)
    }

    fn evaluate_node(node: &Node, features: &[Decimal]) -> Result<Decimal, InterpreterError> {
        match &node.primitive {
            // --- Arity 2 ---
            Primitive::Add => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                Ok(left + right)
            }
            Primitive::Sub => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                Ok(left - right)
            }
            Primitive::Mul => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                Ok(left * right)
            }
            Primitive::ProtectedDiv => {
                let left = Self::evaluate_node(&node.children[0], features)?;
                let right = Self::evaluate_node(&node.children[1], features)?;
                if right.is_zero() {
                    Ok(Decimal::ONE)
                } else {
                    Ok(left / right)
                }
            }
            // --- Arity 3 ---
            Primitive::IfThenElse => {
                let condition = Self::evaluate_node(&node.children[0], features)?;
                // If condition is "true" (non-zero), execute the first branch.
                if !condition.is_zero() {
                    Self::evaluate_node(&node.children[1], features)
                } else {
                    Self::evaluate_node(&node.children[2], features)
                }
            }
            // --- Arity 1 ---
            Primitive::Neg => {
                let child = Self::evaluate_node(&node.children[0], features)?;
                Ok(-child)
            }
            // --- Terminals ---
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

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_simple_evaluation() {
        let json_strategy = r#"
        {
            "root": {
                "primitive": "Add",
                "children": [
                    {
                        "primitive": { "Feature": 0 },
                        "children": []
                    },
                    {
                        "primitive": { "Feature": 1 },
                        "children": []
                    }
                ]
            }
        }
        "#;

        let strategy = GPStrategy::from_json(json_strategy).unwrap();
        let features = vec![dec!(10.5), dec!(5.5)];
        let result = strategy.evaluate(&features).unwrap();

        assert_eq!(result, dec!(16.0));
    }

    #[test]
    fn test_nested_evaluation() {
        let json_strategy = r#"
        {
            "root": {
                "primitive": "Mul",
                "children": [
                    {
                        "primitive": "Add",
                        "children": [
                            { "primitive": { "Feature": 0 }, "children": [] },
                            { "primitive": { "Feature": 1 }, "children": [] }
                        ]
                    },
                    {
                        "primitive": { "Constant": 2.0 },
                        "children": []
                    }
                ]
            }
        }
        "#;

        let strategy = GPStrategy::from_json(json_strategy).unwrap();
        let features = vec![dec!(3.0), dec!(4.0)];
        let result = strategy.evaluate(&features).unwrap();

        assert_eq!(result, dec!(14.0)); // (3 + 4) * 2
    }
}
```

#### **File: `nexus/src/velocity_core/mod.rs`**
```rust
pub mod l2_handler;
pub mod execution;
pub mod models;

```

#### **File: `nexus/src/velocity_core/execution.rs`**
```rust
use anyhow::Result;
use hex;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use sha2::Sha256;

const BYBIT_TESTNET_API_URL: &str = "https://api-testnet.bybit.com";

// --- Time Sync Structs ---
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BybitTime {
    time_nano: String,
}

#[derive(Deserialize)]
struct TimeResult {
    result: BybitTime,
}

// --- Execution Client ---

pub struct ExecutionClient {
    api_key: String,
    api_secret: String,
    client: Client,
}

impl ExecutionClient {
    pub fn new(api_key: String, api_secret: String) -> Self {
        ExecutionClient {
            api_key,
            api_secret,
            client: Client::new(),
        }
    }

    async fn get_server_time(&self) -> Result<String> {
        let time_res = self
            .client
            .get(format!("{}/v5/market/time", BYBIT_TESTNET_API_URL))
            .send()
            .await?
            .json::<TimeResult>()
            .await?;
        let timestamp_nano = &time_res.result.time_nano;
        Ok(timestamp_nano[..timestamp_nano.len() - 6].to_string())
    }

    pub async fn place_order(&self, order_payload: &Value) -> Result<OrderResponse> {
        let timestamp = self.get_server_time().await?;
        let recv_window = "20000";
        let request_body = serde_json::to_string(order_payload)?;

        let mut signature_payload = String::new();
        signature_payload.push_str(&timestamp);
        signature_payload.push_str(&self.api_key);
        signature_payload.push_str(recv_window);
        signature_payload.push_str(&request_body);

        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())?;
        mac.update(signature_payload.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let url = format!("{}/v5/order/create", BYBIT_TESTNET_API_URL);

        let response = self
            .client
            .post(&url)
            .header("X-BAPI-API-KEY", &self.api_key)
            .header("X-BAPI-TIMESTAMP", &timestamp)
            .header("X-BAPI-SIGN", &signature)
            .header("X-BAPI-RECV-WINDOW", recv_window)
            .header("Content-Type", "application/json")
            .body(request_body)
            .send()
            .await?;

        let response_text = response.text().await?;
        match serde_json::from_str::<OrderResponse>(&response_text) {
            Ok(order_response) => {
                if order_response.ret_code == 0 {
                    Ok(order_response)
                } else {
                    Err(anyhow::anyhow!(
                        "Bybit API Error: {} (ret_code: {})",
                        order_response.ret_msg,
                        order_response.ret_code
                    ))
                }
            }
            Err(e) => Err(anyhow::anyhow!(
                "Failed to decode response body. Error: {}. Raw response: {}",
                e,
                response_text
            )),
        }
    }
}

// --- API Data Structures ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderResponse {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: OrderResult,
    pub ret_ext_info: serde_json::Value,
    pub time: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderResult {
    pub order_id: String,
    pub order_link_id: String,
}
```

#### **File: `nexus/src/velocity_core/l2_handler.rs`**
```rust
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;
use std::str::FromStr;

// --- Order Book Logic ---

pub struct OrderBook {
    pub symbol: String,
    // BTreeMap ensures that prices are always sorted.
    // Bids are sorted high to low, so we use std::cmp::Reverse
    pub bids: BTreeMap<std::cmp::Reverse<Decimal>, Decimal>,
    pub asks: BTreeMap<Decimal, Decimal>,
    pub last_update_id: u64,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        OrderBook {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
        }
    }

    pub fn apply_snapshot(&mut self, bids: &[[String; 2]], asks: &[[String; 2]]) {
        self.bids.clear();
        self.asks.clear();
        for [price, size] in bids {
            if let (Ok(p), Ok(s)) = (Decimal::from_str(price), Decimal::from_str(size)) {
                self.bids.insert(std::cmp::Reverse(p), s);
            }
        }
        for [price, size] in asks {
            if let (Ok(p), Ok(s)) = (Decimal::from_str(price), Decimal::from_str(size)) {
                self.asks.insert(p, s);
            }
        }
    }

    pub fn apply_delta(&mut self, bids: &[[String; 2]], asks: &[[String; 2]]) {
        for [price, size] in bids {
            if let (Ok(p), Ok(s)) = (Decimal::from_str(price), Decimal::from_str(size)) {
                if s == dec!(0) {
                    self.bids.remove(&std::cmp::Reverse(p));
                } else {
                    self.bids.insert(std::cmp::Reverse(p), s);
                }
            }
        }
        for [price, size] in asks {
            if let (Ok(p), Ok(s)) = (Decimal::from_str(price), Decimal::from_str(size)) {
                if s == dec!(0) {
                    self.asks.remove(&p);
                } else {
                    self.asks.insert(p, s);
                }
            }
        }
    }

    pub fn print_top_levels(&self, n: usize) {
        println!("\n--- Order Book for {} ---", self.symbol);
        println!("Bids (Top {}):", n);
        for (price, size) in self.bids.iter().take(n) {
            println!("  Price: {:<10} | Size: {}", price.0, size);
        }
        println!("Asks (Top {}):", n);
        for (price, size) in self.asks.iter().take(n) {
            println!("  Price: {:<10} | Size: {}", price, size);
        }
        println!("-------------------------");
    }

    pub fn calculate_microstructure_features(&self) -> Option<MicrostructureFeatures> {
        let best_bid = self.bids.iter().next();
        let best_ask = self.asks.iter().next();

        if let (Some((bid_price, bid_size)), Some((ask_price, ask_size))) = (best_bid, best_ask) {
            // 1. Top-of-Book Imbalance (Proxy for OFI)
            let total_top_volume = *bid_size + *ask_size;
            let top_of_book_imbalance = if total_top_volume > dec!(0) {
                *bid_size / total_top_volume
            } else {
                dec!(0.5) // Neutral if no volume
            };

            // 2. Book Pressure (Top 5 Levels)
            let bid_volume_top5: Decimal = self.bids.iter().take(5).map(|(_, size)| size).sum();
            let ask_volume_top5: Decimal = self.asks.iter().take(5).map(|(_, size)| size).sum();
            let total_volume_top5 = bid_volume_top5 + ask_volume_top5;
            let book_pressure = if total_volume_top5 > dec!(0) {
                bid_volume_top5 / total_volume_top5
            } else {
                dec!(0.5) // Neutral if no volume
            };

            // 3. Close Location Value (CLV) Proxy
            // True CLV requires the last trade price: (last_trade - best_bid) / (best_ask - best_bid)
            // We use the mid-price as a proxy, which will always be ~0.5.
            // This is a placeholder for when trade data might be integrated.
            let mid_price = (bid_price.0 + ask_price) / dec!(2);
            let spread = *ask_price - bid_price.0;
            let clv_proxy = if spread > dec!(0) {
                (mid_price - bid_price.0) / spread
            } else {
                dec!(0.5)
            };

            return Some(MicrostructureFeatures {
                top_of_book_imbalance,
                book_pressure,
                clv_proxy,
            });
        }
        None
    }
}

#[derive(Debug)]
pub struct MicrostructureFeatures {
    /// Ratio of volume at the best bid vs total volume at best bid/ask. > 0.5 means more bid volume.
    pub top_of_book_imbalance: Decimal,
    /// Ratio of total bid volume vs total volume in the top 5 levels. > 0.5 means more bid-side pressure.
    pub book_pressure: Decimal,
    /// Proxy for Close Location Value, using mid-price. Will be ~0.5.
    pub clv_proxy: Decimal,
}
```

#### **File: `nexus/src/velocity_core/models.rs`**
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderBookData {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
    #[serde(rename = "u")]
    pub update_id: u64,
    pub seq: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BybitResponse {
    Success(SuccessResponse),
    OrderBook(OrderBookResponse),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SuccessResponse {
    pub success: bool,
    pub ret_msg: String,
    pub conn_id: String,
    pub op: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OrderBookResponse {
    pub topic: String,
    #[serde(rename = "type")]
    pub response_type: String, // "snapshot" or "delta"
    #[serde(rename = "ts")]
    pub timestamp: u64,
    #[serde(rename = "cts")]
    pub cross_timestamp: u64,
    #[serde(rename = "data")]
    pub data: OrderBookData,
}

#[derive(Debug, Serialize)]
pub struct SubscriptionMessage<'a> {
    pub op: &'a str,
    pub args: Vec<String>,
}

```

### 2. The `forge` Directory (Python R&D)

#### **File: `forge/requirements.txt`**
```
deap
lightgbm
tigramite
numpy
cma

```

#### **File: `forge/gp_framework.py`**
```python
import operator
import random
import numpy

from deap import algorithms, base, creator, tools, gp

# --- 1. Define the Primitive Set ---
# These are the building blocks for our GP trees.
# We'll start with a simple set and expand it later with trading indicators.

# The primitive set will take 3 arguments (inputs)
pset = gp.PrimitiveSet("MAIN", 3) 

# Basic arithmetic operations
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)

# A protected division to avoid division by zero errors
def protectedDiv(left, right):
    with numpy.errstate(divide='ignore',invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x
pset.addPrimitive(protectedDiv, 2)


# Conditional logic
def if_then_else(input, output1, output2):
    return output1 if input else output2
pset.addPrimitive(if_then_else, 3)

# Rename arguments for clarity
pset.renameArguments(ARG0='imbalance', ARG1='pressure', ARG2='clv')


# --- 2. Create the Types ---
# We need to define the structure of our individuals (the GP trees) and the fitness.

# We want to maximize a single objective (e.g., profit), so we use a maximizing fitness.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Each individual is a GP tree, with the FitnessMax attribute we just defined.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# --- 3. Set up the Toolbox ---
# The toolbox is where we register the functions (operators) for our evolution.

toolbox = base.Toolbox()

# Attribute generators: how to create expressions and trees
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operator registration: the core evolutionary operators
toolbox.register("compile", gp.compile, pset=pset)

# This is a placeholder evaluation function.
# In the next step, this will be replaced by our TTT fitness function
# that interacts with the Rust simulator.
from fitness_ttt import evaluate_individual

# A dummy DSG for testing. In a real run, this would be the output of the Chimera Engine.
dummy_dsg = [0.8, 0.5] # [order_prob, aggressiveness]

toolbox.register("evaluate", evaluate_individual, pset=pset, dsg=dummy_dsg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorators to limit the height of the trees, preventing "bloat"
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# --- Example Usage ---
def main():
    """
    A simple example of how to run the GP framework.
    """
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    # This is a standard DEAP evolutionary algorithm.
    # It will run for a few generations and print the stats.
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 5, stats=stats, halloffame=hof)

    print("\n--- Genetic Programming Run Complete ---")
    print("Best individual found:")
    print(hof[0])
    
    return pop, stats, hof

if __name__ == "__main__":
    main()
```

#### **File: `forge/fitness_ttt.py`**
```python
import sys
import os
import random
import operator
from deap import gp

# --- Add the Rust module to the Python path ---
module_path = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "release")
sys.path.append(module_path)

try:
    import nexus
except ImportError as e:
    print(f"Failed to import 'nexus' module: {e}")
    sys.exit(1)

# --- Agents ---

class TTTAgent:
    """ The 'Apex Predator' agent, controlled by a GP tree. """
    def __init__(self, agent_id, strategy_func):
        self.id = agent_id
        self.strategy_func = strategy_func
        self.pnl = 0.0

    def on_tick(self, market_state):
        mock_feature_0 = random.random()
        mock_feature_1 = random.random()
        signal = self.strategy_func(mock_feature_0, mock_feature_1)
        
        orders = []
        if signal > 0.5 and market_state.asks:
            best_ask = min(market_state.asks.keys())
            orders.append(nexus.Order(self.id, best_ask, 0.01, True))
        elif signal < -0.5 and market_state.bids:
            best_bid = max(market_state.bids.keys())
            orders.append(nexus.Order(self.id, best_bid, 0.01, False))
        return orders

class AdversarialAgent:
    """ The 'competition' agent, controlled by the Dominant Strategy Genome. """
    def __init__(self, agent_id, genome):
        self.id = agent_id
        self.order_prob = genome[0]
        self.aggressiveness = genome[1]

    def on_tick(self, market_state):
        orders = []
        if random.random() < self.order_prob:
            mid_price = 100.0
            if market_state.bids and market_state.asks:
                mid_price = (max(market_state.bids.keys()) + min(market_state.asks.keys())) / 2.0
            
            spread = self.aggressiveness * 0.1
            buy_price = mid_price - spread
            sell_price = mid_price + spread
            
            orders.append(nexus.Order(self.id, buy_price, 0.1, True))
            orders.append(nexus.Order(self.id, sell_price, 0.1, False))
        return orders

# --- The Main Evaluation Function (Hybrid Crucible) ---

def evaluate_individual(individual, pset, dsg):
    """
    Evaluates a GP individual against the competition defined by the DSG.
    """
    strategy_func = gp.compile(expr=individual, pset=pset)
    sim = nexus.Simulation()
    
    # 1. Add the Apex Predator agent (our GP individual)
    apex_predator = TTTAgent(agent_id=1, strategy_func=strategy_func)
    sim.add_agent(apex_predator)
    
    # 2. Add the competition agent, configured with the DSG
    competition = AdversarialAgent(agent_id=2, genome=dsg)
    sim.add_agent(competition)

    # 3. Run the simulation
    num_ticks = 200
    total_pnl = 0.0
    
    for _ in range(num_ticks):
        trades = sim.tick()
        # Calculate PnL for our Apex Predator
        for trade in trades:
            if trade.aggressor_agent_id == 1: # Our agent was the aggressor
                # This is a simplified PnL calculation. A real one would be more complex.
                # It doesn't properly track position, just immediate trade profit/loss.
                market_state = sim.get_market_state()
                if market_state.bids and market_state.asks:
                    mid_price = (max(market_state.bids.keys()) + min(market_state.asks.keys())) / 2.0
                    # If we bought, profit is mid_price - trade.price
                    # If we sold, profit is trade.price - mid_price
                    # This logic is flawed, but serves as a placeholder.
                    pnl = (mid_price - trade.price) if trade.size > 0 else (trade.price - mid_price)
                    total_pnl += pnl

    return (total_pnl,)


# --- Example Usage ---
if __name__ == '__main__':
    # A dummy DSG for testing
    dummy_dsg = [0.8, 0.5] # High probability, medium aggressiveness
    
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.addPrimitive(operator.add, 2)
    dummy_individual = gp.PrimitiveTree.from_string("add(ARG0, ARG1)", pset)
    
    print("--- Evaluating a dummy individual in the Hybrid Crucible ---")
    fitness = evaluate_individual(dummy_individual, pset, dummy_dsg)
    print(f"Adversarial Fitness of dummy individual: {fitness[0]}")

```

#### **File: `forge/chimera_engine.py`**
```python
import cma
import numpy as np
import sys
import os

# --- Add the Rust module to the Python path ---
module_path = os.path.join(os.path.dirname(__file__), "..", "nexus", "target", "release")
sys.path.append(module_path)

try:
    import nexus
except ImportError as e:
    print(f"Failed to import 'nexus' module: {e}")
    sys.exit(1)

# --- The Adversarial Agent ---

class AdversarialAgent:
    """
    A simple agent whose behavior is defined by a 'genome' of parameters.
    This agent represents the "competition" in the market.
    """
    def __init__(self, agent_id, genome):
        self.id = agent_id
        # Genome parameters could be things like:
        # genome[0]: probability of submitting an order
        # genome[1]: aggressiveness (how far from the mid-price to place orders)
        self.order_prob = genome[0]
        self.aggressiveness = genome[1]

    def on_tick(self, market_state):
        orders = []
        if np.random.rand() < self.order_prob:
            # Simple logic: place a buy and a sell order around a central price
            mid_price = 100.0 # Placeholder
            if market_state.bids and market_state.asks:
                mid_price = (max(market_state.bids.keys()) + min(market_state.asks.keys())) / 2.0
            
            spread = self.aggressiveness * 0.1
            buy_price = mid_price - spread
            sell_price = mid_price + spread
            
            orders.append(nexus.Order(self.id, buy_price, 0.1, True)) # Bid
            orders.append(nexus.Order(self.id, sell_price, 0.1, False)) # Ask
            
        return orders

# --- The Objective Function for CMA-ES ---

def objective_function(genome):
    """
    This function takes a genome, runs a simulation with an agent configured by that genome,
    and returns a score. The goal of CMA-ES is to find the genome that minimizes this score.
    """
    sim = nexus.Simulation()
    
    # Add the adversarial agent with the given genome
    adversary = AdversarialAgent(agent_id=1, genome=genome)
    sim.add_agent(adversary)
    
    total_volume = 0.0
    
    # Run the simulation
    for _ in range(100): # Short simulation for demonstration
        trades = sim.tick()
        for trade in trades:
            total_volume += trade.size
            
    # The objective is to maximize volume, so we return the negative volume
    # because CMA-ES minimizes by default.
    return -total_volume

# --- Main Chimera Engine Logic ---

def main():
    """
    Sets up and runs the CMA-ES optimizer to find the Dominant Strategy Genome (DSG).
    """
    # Initial guess for the genome and standard deviation
    initial_genome = [0.5, 0.5]  # [order_prob, aggressiveness]
    initial_std_dev = 0.2
    
    # We use the 'fmin' function from the cma library
    # It will iteratively call our objective_function to find the best genome.
    best_genome, es = cma.fmin2(objective_function, initial_genome, initial_std_dev, {'bounds': [0, 1]})
    
    print("\n--- Chimera Engine Run Complete ---")
    print("Optimal Genome (DSG) found:")
    print(f"  - Order Probability: {best_genome[0]}")
    print(f"  - Aggressiveness: {best_genome[1]}")
    
    # You can also get more details from the evolution strategy object
    # es.result_pretty()

if __name__ == "__main__":
    main()

```

#### **File: `forge/serialize_strategy.py`**
```python
import json
from deap import gp

def _recursive_builder(individual, index):
    """
    A recursive helper function to build the JSON tree.
    Returns the node and the index of the next node to process.
    """
    node_primitive = individual[index]
    
    if isinstance(node_primitive, gp.Terminal):
        if node_primitive.name.startswith("ARG"):
            primitive = {"Feature": int(node_primitive.name[3:])}
        else:
            primitive = {"Constant": node_primitive.value}
        return {"primitive": primitive, "children": []}, index + 1
    else: # It's a Primitive
        children = []
        current_index = index + 1
        for _ in range(node_primitive.arity):
            child_node, next_index = _recursive_builder(individual, current_index)
            children.append(child_node)
            current_index = next_index
            
        return {"primitive": node_primitive.name, "children": children}, current_index

def deap_to_json(individual):
    """
    Converts a DEAP GP individual into a JSON string for the Rust interpreter.
    """
    root_node, _ = _recursive_builder(individual, 0)
    return json.dumps({"root": root_node}, indent=4)


# --- Example Usage ---
if __name__ == '__main__':
    import operator

    # 1. Create a dummy DEAP individual
    pset = gp.PrimitiveSet("MAIN", 3)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.renameArguments(ARG0='imbalance', ARG1='pressure', ARG2='clv')
    
    # Create a simple example expression
    expr_str = "add(imbalance, pressure)"
    individual = gp.PrimitiveTree.from_string(expr_str, pset)

    # 2. Convert it to JSON
    json_output = deap_to_json(individual)

    # 3. Print and save the result
    print("--- DEAP Individual ---")
    print(individual)
    
    print("\n--- Serialized JSON Strategy ---")
    print(json_output)

    # Save to a file for the Rust engine to use
    with open("strategy.json", "w") as f:
        f.write(json_output)
        
    print("\nSaved strategy to 'strategy.json'")

```

#### **File: `forge/sae_oracle.py`**
```python
# Placeholder for the Surrogate-Assisted Evolution (SAE) model.
# This will use LightGBM to create a "Fitness Oracle" that can quickly
# estimate the fitness of a large number of individuals without running the
# full, expensive simulation for each one.

```

#### **File: `forge/strategy.json`**
```json
{
    "root": {
        "primitive": "add",
        "children": [
            {
                "primitive": {
                    "Feature": 0
                },
                "children": []
            },
            {
                "primitive": {
                    "Feature": 1
                },
                "children": []
            }
        ]
    }
}
```

### 3. Root Directory

#### **File: `test_bridge.py`**
```python
import sys
import os

# Add the directory containing the .pyd file to the Python path
# This is necessary so Python can find our Rust module.
# The path is relative to this script's location.
module_path = os.path.join(os.path.dirname(__file__), "nexus", "target", "release")
sys.path.append(module_path)

try:
    import nexus
    print("Successfully imported 'nexus' module from Rust.")
except ImportError as e:
    print(f"Failed to import 'nexus' module: {e}")
    print(f"Please ensure '{os.path.join(module_path, 'nexus.pyd')}' exists.")
    sys.exit(1)


class MyPythonAgent:
    """
    A simple agent defined in Python that implements the required `id` and `on_tick` methods.
    """
    def __init__(self, agent_id):
        self.id = agent_id
        self.tick_count = 0

    def on_tick(self, market_state):
        """
        This method is called from the Rust simulation on every tick.
        """
        print(f"[Python Agent {self.id}] Tick {self.tick_count}. Market sequence: {market_state.sequence}")
        print(f"[Python Agent {self.id}] Bids: {market_state.bids}, Asks: {market_state.asks}")
        
        self.tick_count += 1
        
        # On the first tick, submit a buy order.
        if self.tick_count == 1:
            print(f"[Python Agent {self.id}] Submitting a buy order.")
            # nexus.Order(agent_id, price, size, is_bid)
            return [nexus.Order(self.id, 100.0, 5.0, True)]
            
        # On the third tick, submit a sell order that should match the first order.
        if self.tick_count == 3:
            print(f"[Python Agent {self.id}] Submitting a sell order.")
            return [nexus.Order(self.id, 100.0, 2.0, False)]

        # On all other ticks, do nothing.
        return []


def main():
    print("\n--- Setting up simulation ---")
    # Create a simulation instance from our Rust code
    sim = nexus.Simulation()

    # Create two Python agents
    agent1 = MyPythonAgent(agent_id=1)
    agent2 = MyPythonAgent(agent_id=2)

    # Add the Python agents to the Rust simulation
    sim.add_agent(agent1)
    sim.add_agent(agent2)
    print("Added 2 Python agents to the Rust simulation.")

    print("\n--- Running simulation for 5 ticks ---")
    for i in range(5):
        print(f"\n--- Tick {i+1} ---")
        trades = sim.tick()
        if trades:
            print(f"Trades occurred in tick {i+1}:")
            for trade in trades:
                print(f"  - Price: {trade.price}, Size: {trade.size}, Aggressor: {trade.aggressor_agent_id}, Resting: {trade.resting_agent_id}")
        else:
            print("No trades occurred.")
            
    print("\n--- Simulation finished ---")
    final_market_state = sim.get_market_state()
    print(f"Final Bids: {final_market_state.bids}")
    print(f"Final Asks: {final_market_state.asks}")


if __name__ == "__main__":
    main()

```

---

## II. Roadmap to 100% Completion

# Roadmap to 100% Completion: The Apex Singularity Nexus

This document outlines the remaining steps required to evolve the Apex Singularity Nexus (ASN) from its current functional skeleton to the fully autonomous, intelligent, and adaptive trading system envisioned in the original blueprint.

The core architecture is in place. What follows is the implementation of the sophisticated logic and data-driven "brains" that will inhabit this architecture.

---

### **Phase 1: Enhancing the Core Logic (Bridging the Gaps)**

This phase focuses on replacing placeholder logic with production-ready components.

#### **1.1. Dynamic Order Sizing (GP-Controlled Quantity)**

*   **Objective:** Remove the hardcoded order quantity and allow the evolved strategy to determine its own size.
*   **Tasks:**
    *   **[ ] Python (`gp_framework.py`):**
        *   Modify the `creator` to define an individual's fitness with two objectives (e.g., `weights=(1.0, 0.1,)` for PnL and a small weight for a secondary objective if needed).
        *   Modify the `creator` to define an `Individual` that outputs a list or tuple of values, not a single float. The GP tree will now have two root nodes or a single root that returns a tuple.
    *   **[ ] Python (`fitness_ttt.py`):**
        *   Update the `TTTAgent`'s `on_tick` method. It will now receive two values from the `strategy_func`: `signal` and `size`.
        *   The `size` value should be clamped or transformed (e.g., via a sigmoid function) to a reasonable range (e.g., 0.01 to 0.5 BTC) to prevent rogue orders.
        *   The `nexus.Order` call will now use this dynamic `size` instead of a hardcoded value.
    *   **[ ] Rust (`lib.rs` & `gp_interpreter.rs`):**
        *   The Rust interpreter's `evaluate` function must be updated to return a tuple or a struct containing both the signal and the size. This may require adjusting the `Node` and `Primitive` enums to support multiple output types.

#### **1.2. True Time-to-Target (TTT) Fitness**

*   **Objective:** Implement the "Velocity Fitness" as the primary objective function, replacing the placeholder PnL.
*   **Tasks:**
    *   **[ ] Python (`fitness_ttt.py`):**
        *   Modify the `TTTAgent` to track its equity over time within the simulation.
        *   Define a capital target (e.g., 2x initial capital).
        *   The `evaluate_individual` function will now run the simulation until the agent either hits the capital target or a maximum time limit is reached.
        *   **The fitness score will be the inverse of the number of ticks it took to reach the target.** A faster time equals a higher fitness. If the target is not reached, a penalty should be applied.

---

### **Phase 2: Implementing the Full ACN Synthesis**

This phase focuses on building out the sophisticated, data-driven logic for the Chimera Engine and Hybrid Crucible.

#### **2.1. Chimera Engine: Microstructure Mimicry**

*   **Objective:** Evolve the Dominant Strategy Genome (DSG) by forcing the simulation to replicate the statistical properties of real historical market data.
*   **Tasks:**
    *   **[ ] Data Pipeline (Python):**
        *   Implement a script to download and preprocess historical L2 order book data (e.g., from a provider like Kaiko, or by capturing it from the live feed). This data should be stored in an efficient format (e.g., Parquet).
    *   **[ ] Feature Extraction (Python):**
        *   Create a function that calculates a "feature vector" from a segment of market data (real or simulated). This vector should contain key statistical properties:
            *   Volatility of the mid-price.
            *   Average spread.
            *   Autocorrelation of returns.
            *   Order book imbalance distribution.
            *   Trade volume distribution.
    *   **[ ] Objective Function (`chimera_engine.py`):**
        *   Replace the current `objective_function` (which just maximizes volume).
        *   The new function will:
            1.  Load a chunk of real historical data and calculate its feature vector (the "target vector").
            2.  Run the Rust simulation with an `AdversarialAgent` configured by the input `genome`.
            3.  Capture the simulated market data and calculate its feature vector.
            4.  **The fitness score will be the Euclidean distance (or similar metric) between the target vector and the simulated vector.** The CMA-ES algorithm will minimize this distance, forcing the simulation to become statistically identical to reality.

#### **2.2. Hybrid Crucible: The Dual-Fitness Function**

*   **Objective:** Implement the complete Dual-Fitness Function, combining adversarial performance with performance on historical data.
*   **Tasks:**
    *   **[ ] Causal Fitness Backtester (Rust):**
        *   Create a new, simplified backtester in Rust. This backtester will not be agent-based. It will be a fast, vectorized engine that replays historical data.
        *   It will take a `GPStrategy` and a historical dataset as input.
        *   It will loop through the historical data, feed the features to the strategy, and simulate the execution of the resulting signals, calculating the final PnL or TTT.
    *   **[ ] PyO3 Bindings (Rust):**
        *   Expose the new Causal Fitness backtester to Python via PyO3.
    *   **[ ] Dual-Fitness (`fitness_ttt.py`):**
        *   The `evaluate_individual` function will be updated to calculate two scores:
            1.  **Adversarial Fitness:** The existing TTT score from competing against the DSG in the HF-ABM.
            2.  **Causal Fitness:** The TTT score from running the strategy in the new, fast Rust backtester with historical data.
        *   The final fitness returned to the GP algorithm will be a weighted average of these two scores.

---

### **Phase 3: Advanced Intelligence & Live Operations**

This phase implements the final, most advanced features of the blueprint.

#### **3.1. Causal Discovery (`Tigramite`) Integration**

*   **Objective:** Use causal discovery to select only the most robust features to be used by the GP algorithm.
*   **Tasks:**
    *   **[ ] Causal Analysis Script (Python):**
        *   Create a new script that loads historical data.
        *   Uses `Tigramite` (the PCMCI algorithm) to generate a causal graph of the available microstructure features.
    *   **[ ] Dynamic Primitive Set (`gp_framework.py`):**
        *   Modify the GP framework to dynamically generate its `PrimitiveSet`.
        *   Before the evolutionary run starts, it will call the Causal Analysis script.
        *   **Only the features that are identified as direct causal parents of the target variable (e.g., future returns) will be included in the `PrimitiveSet`** (e.g., `pset.renameArguments(ARG0='imbalance', ARG1='clv')` if `pressure` was found to be non-causal).

#### **3.2. The Full Crucible (Live Operations in Rust)**

*   **Objective:** Implement the full logic for managing 6 concurrent models, hot-swapping, and the "$200 Sudden Death" rule.
*   **Tasks:**
    *   **[ ] `main.rs` Refactoring:**
        *   Create a `TradingModel` struct that encapsulates a `GPStrategy`, an `OrderBook`, and its current capital.
        *   Instantiate a vector or array of 6 `TradingModel` instances.
    *   **[ ] Concurrent Execution:**
        *   Modify the main loop to iterate through all 6 models on each tick.
        *   Each model will receive the relevant market data, evaluate its unique strategy, and make its own trading decisions.
    *   **[ ] The Watchtower ($200 Rule):**
        *   Implement logic to track the PnL of each `TradingModel`.
        *   If a model's capital drops to zero, its loop is terminated, and a "liquidation" order is sent to close any open positions.
    *   **[ ] The Pit Crew (Hot-Swapping):**
        *   Implement a file watcher that monitors a directory for new `strategy_challenger_*.json` files.
        *   When a new challenger file appears, load it into a new `TradingModel` that runs in "shadow mode" (it makes decisions but does not send orders).
        *   Track the virtual performance of the challenger. If it consistently outperforms the live champion for that asset, atomically swap the live `GPStrategy` with the challenger's strategy.

This roadmap provides a clear and detailed path to achieving the true 100% vision of this blueprint.