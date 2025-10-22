use anyhow::Result;
use nexus::{
    inference::gp_interpreter::{GPStrategy, StrategyOutput},
    velocity_core::{
        execution::{private_ws::{self, PrivateOrderFill}, ExecutionClient},
        iel::{ExecutionAction, IntelligentExecutor},
        l2_handler::OrderBook,
        models::{BybitResponse, SubscriptionMessage},
    },
};
use rust_decimal::prelude::{FromStr, ToPrimitive};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::connect_async;
use futures_util::{StreamExt, SinkExt};
use serde_json::json;
use notify::{RecommendedWatcher, RecursiveMode, Watcher, EventKind};
use parking_lot::RwLock;
use uuid::Uuid;

const IEL_MODEL_PATH: &str = "iel_model.onnx";

const BYBIT_WS_URL: &str = "wss://stream-testnet.bybit.com/v5/public/spot";
const STRATEGY_DIR: &str = "../forge/";
const INITIAL_CAPITAL: Decimal = dec!(200.0);
const PERFORMANCE_THRESHOLD: f64 = 1.10; // Challenger must be 10% better
const MIN_SHADOW_EVAL_TICKS: u64 = 600; // Minimum ticks before considering a swap (e.g., 10 mins)

// --- Trading Model Structure ---

#[derive(Debug, Clone, Copy, PartialEq)]
enum ModelState {
    Live,
    Shadow,
    Terminated,
}

struct TradingModel {
    id: u64,
    symbol: String,
    slot_id: u8, // Identifies which of the 2 slots this model occupies (0 or 1)
    strategy: Arc<RwLock<GPStrategy>>,
    capital: Decimal,
    inventory: Decimal,
    state: ModelState,
    // Performance Tracking
    realized_pnl: Decimal,
    eval_start_time: Instant,
    eval_ticks: u64,
    cumulative_loss: f64, // NEW: For Hedge Algorithm
}

impl TradingModel {
    fn new(id: u64, symbol: String, slot_id: u8, strategy: GPStrategy, state: ModelState) -> Self {
        Self {
            id, symbol, slot_id,
            strategy: Arc::new(RwLock::new(strategy)),
            capital: INITIAL_CAPITAL,
            inventory: dec!(0),
            state,
            realized_pnl: dec!(0),
            eval_start_time: Instant::now(),
            eval_ticks: 0,
            cumulative_loss: 0.0, // NEW
        }
    }

    fn calculate_equity(&self, mid_price: Option<Decimal>) -> Decimal {
        let inventory_value = mid_price.unwrap_or(dec!(0)) * self.inventory;
        self.capital + inventory_value
    }

    // The Watchtower check ($200 Sudden Death Rule)
    // Returns true if liquidation is needed
    fn check_sudden_death(&mut self, mid_price: Option<Decimal>) -> bool {
        if self.state == ModelState::Terminated { return false; }

        let equity = self.calculate_equity(mid_price);
        
        if equity <= dec!(0) {
            println!("\n[WATCHTOWER] SUDDEN DEATH! M{} ({}) terminated. Equity: ${:.2}\n", self.id, self.symbol, equity);
            self.state = ModelState::Terminated;
            // Needs liquidation if inventory is not zero
            return self.inventory.abs() > dec!(0.00001);
        }
        false
    }

    // Performance metric (PnL per tick)
    fn get_performance_score(&self) -> f64 {
        if self.eval_ticks > 0 {
            // Convert Decimal PnL to f64 for comparison
            self.realized_pnl.to_f64().unwrap_or(0.0) / self.eval_ticks as f64
        } else {
            0.0
        }
    }
}

// --- Global State (The Crucible) ---
struct GlobalState {
    order_books: HashMap<String, OrderBook>,
    models: HashMap<u64, TradingModel>,
    symbol_map: HashMap<String, Vec<u64>>,
    // Track the current Champion ID for each specific slot (Symbol, SlotID)
    champions: HashMap<(String, u8), u64>, 
    model_id_counter: u64,
}


// --- Pit Crew (Strategy Hot-Swapping and Evaluation) ---

// Task 1: Intake (File Watcher)
async fn start_pit_crew_intake(state: Arc<Mutex<GlobalState>>) -> Result<()> {
    // ... (Implementation identical to the previous iteration: Watches directory, loads new strategies into Shadow Mode for all applicable slots)
    // ... (Omitted for brevity, ensure the logic from Part 2 implementation is included here)
}

// Task 2: Evaluation (Periodic Performance Review)
async fn start_pit_crew_evaluation(state: Arc<Mutex<GlobalState>>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60)); // Evaluate every 60s

    loop {
        interval.tick().await;
        println!("[PIT CREW EVAL] Running performance review...");
        let mut guard = state.lock().await;

        // Iterate over each Champion slot
        let champion_slots = guard.champions.keys().cloned().collect::<Vec<_>>();
        
        for (symbol, slot_id) in champion_slots {
            if let Some(&champion_id) = guard.champions.get(&(symbol.clone(), slot_id)) {
                
                // 1. Get Champion Performance
                let champion_score = guard.models.get(&champion_id).map(|m| m.get_performance_score()).unwrap_or(0.0);

                // 2. Find the Best Challenger
                let mut best_challenger_id = None;
                let mut best_challenger_score = -f64::INFINITY;

                if let Some(model_ids) = guard.symbol_map.get(&symbol) {
                    for &id in model_ids {
                        if let Some(model) = guard.models.get(&id) {
                            // Check criteria: Shadow state, correct slot, minimum eval time
                            if model.state == ModelState::Shadow && model.slot_id == slot_id && model.eval_ticks >= MIN_SHADOW_EVAL_TICKS {
                                let score = model.get_performance_score();
                                if score > best_challenger_score {
                                    best_challenger_score = score;
                                    best_challenger_id = Some(id);
                                }
                            }
                        }
                    }
                }

                // 3. The Swap Decision
                if let Some(challenger_id) = best_challenger_id {
                    // Ensure the challenger is significantly better AND has positive performance.
                    if best_challenger_score > 0.0 && best_challenger_score > champion_score * PERFORMANCE_THRESHOLD {
                        println!("[PIT CREW SWAP] Hot-Swapping {} Slot {}. Promoting M{}, Demoting M{} ({:.6} vs {:.6})", 
                            symbol, slot_id, challenger_id, champion_id, best_challenger_score, champion_score);
                        
                        // Execute the swap
                        if let Some(champion) = guard.models.get_mut(&champion_id) {
                            champion.state = ModelState::Shadow; // Demote champion
                            // Reset metrics
                            champion.eval_ticks = 0;
                            champion.realized_pnl = dec!(0);
                        }
                        if let Some(challenger) = guard.models.get_mut(&challenger_id) {
                            challenger.state = ModelState::Live; // Promote challenger
                             // Reset metrics
                            challenger.eval_ticks = 0;
                            challenger.realized_pnl = dec!(0);
                        }
                        guard.champions.insert((symbol.clone(), slot_id), challenger_id);
                    }
                }
            }
        }
    }
}


// --- Continuous Weight Optimization (CWO) ---
async fn start_cwo_rebalance(state: Arc<Mutex<GlobalState>>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60)); // Rebalance every 60s
    const ETA: f64 = 0.5; // Aggressive learning rate as per blueprint

    loop {
        interval.tick().await;
        println!("[CWO] Running capital rebalance using Hedge Algorithm...");
        let mut guard = state.lock().await;

        // Group champions by symbol
        let mut champions_by_symbol: HashMap<String, Vec<u64>> = HashMap::new();
        for ((symbol, _), champion_id) in &guard.champions {
            if let Some(model) = guard.models.get(champion_id) {
                if model.state == ModelState::Live {
                    champions_by_symbol.entry(symbol.clone()).or_default().push(*champion_id);
                }
            }
        }

        for (symbol, champion_ids) in champions_by_symbol {
            if champion_ids.len() == 2 {
                let id1 = champion_ids[0];
                let id2 = champion_ids[1];

                let loss1 = guard.models.get(&id1).map(|m| m.cumulative_loss).unwrap_or(0.0);
                let loss2 = guard.models.get(&id2).map(|m| m.cumulative_loss).unwrap_or(0.0);

                // Hedge Algorithm: weight = exp(-eta * cumulative_loss)
                let weight1_unscaled = (-ETA * loss1).exp();
                let weight2_unscaled = (-ETA * loss2).exp();
                let total_weight = weight1_unscaled + weight2_unscaled;

                let weight1 = if total_weight > 0.0 { weight1_unscaled / total_weight } else { 0.5 };
                let weight2 = if total_weight > 0.0 { weight2_unscaled / total_weight } else { 0.5 };

                // Get total capital for the symbol pair to reallocate
                let total_capital = guard.models.get(&id1).unwrap().capital + guard.models.get(&id2).unwrap().capital;

                // Reallocate capital
                if let Some(model1) = guard.models.get_mut(&id1) {
                    model1.capital = total_capital * Decimal::from_f64(weight1).unwrap_or(dec!(0.5));
                }
                if let Some(model2) = guard.models.get_mut(&id2) {
                    model2.capital = total_capital * Decimal::from_f64(weight2).unwrap_or(dec!(0.5));
                }
                 println!("[CWO] Rebalanced {}. M{}: {:.2}% (Loss: {:.4}), M{}: {:.2}% (Loss: {:.4})", 
                    symbol, id1, weight1 * 100.0, loss1, id2, weight2 * 100.0, loss2);
            }
        }
    }
}


// --- Main Execution ---

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- ASN NEXUS INITIALIZING (Crucible v3) ---");

    // --- 1. Initialize Core Components ---
    let api_key = env::var("BYBIT_API_KEY").expect("BYBIT_API_KEY must be set");
    let api_secret = env::var("BYBIT_API_SECRET").expect("BYBIT_API_SECRET must be set");
    
    let mut execution_client_instance = ExecutionClient::new(api_key.clone(), api_secret.clone());
    
    let iel_model_path = format!("{}iel_model.onnx", STRATEGY_DIR);
    if std::path::Path::new(&iel_model_path).exists() {
        if let Err(e) = execution_client_instance.load_iel_model(&iel_model_path) {
             println!("[IEL] WARNING: Failed to initialize IEL session: {}. Execution will use default tactics.", e);
        }
    } else {
        println!("[IEL] INFO: iel_model.onnx not found at {}. Run forge/iel_trainer.py first.", iel_model_path);
    }

    let execution_client = Arc::new(execution_client_instance);

    // Create a channel for inventory updates
    let (tx, mut rx) = mpsc::channel::<PrivateOrderFill>(128);

    // (Default strategy loading logic remains the same - ensure it handles 5 features)
    // ...

    // --- 2. Initialize Global State and Models ---
    let mut state = GlobalState { /* ... Initialization ... */ };

    let assets = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for asset in &assets {
        // Initialize OrderBook (Now handles MTFA history)
        state.order_books.insert(asset.to_string(), OrderBook::new(asset.to_string()));
        // ... (Initialize 2 Champions per asset)
    }

    let shared_state = Arc::new(Mutex::new(state));

    // --- 3. Start Auxiliary Tasks ---
    
    // Start Pit Crew Intake (File Watcher)
    let intake_state = shared_state.clone();
    tokio::spawn(async move {
        start_pit_crew_intake(intake_state).await.ok();
    });

    // Start Pit Crew Evaluation Runner
    let eval_state = shared_state.clone();
    tokio::spawn(async move {
        start_pit_crew_evaluation(eval_state).await;
    });

    // Start CWO Rebalance Runner
    let cwo_state = shared_state.clone();
    tokio::spawn(async move {
        start_cwo_rebalance(cwo_state).await;
    });

    // Start Private WebSocket Stream
    let api_key_clone = api_key.clone();
    let api_secret_clone = api_secret.clone();
    tokio::spawn(async move {
        loop {
            println!("[PRIVATE STREAM] Attempting to connect...");
            if let Err(e) = private_ws::connect(&api_key_clone, &api_secret_clone, tx.clone()).await {
                eprintln!("[PRIVATE STREAM] Connection error: {}. Reconnecting in 5s...", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    });


    // --- 4. Connect to Public WebSocket and Subscribe ---
    // (Public WebSocket connection and subscription logic)
    // ...

    println!("\n--- NEXUS IS LIVE (Crucible v3) ---");

    // --- 5. Main Event Loop (Public Stream Processing) ---
    loop {
        tokio::select! {
            Some(msg) = read_public.next() => {
                let arrival_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros();

                 // (Message error handling)

                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                    if let Ok(BybitResponse::OrderBook(resp)) = serde_json::from_str::<BybitResponse>(&text) {
                        
                        // --- INGESTION LATENCY LOGIC ---
                        let bybit_ts = resp.ts;
                        println!("[LATENCY] Ingestion: (Now_us: {}, Bybit_ms: {})", arrival_time, bybit_ts);

                        let symbol = resp.data.symbol.clone();
                        let mut state_guard = shared_state.lock().await;
                        
                        // ... (rest of the loop)
                        
                                                        if model.state == ModelState::Live {
                                                            // LIVE EXECUTION
                                                            let m_id = model.id;
                                                            let order_link_id = format!("asn_M{}_{}", model.id, Uuid::new_v4().simple());

                                                            let base_payload = json!({
                                                                "category": "spot", "symbol": &model.symbol, "side": side,
                                                                "qty": qty.to_string(),
                                                                "orderLinkId": order_link_id
                                                            });

                                                            let client_clone = execution_client.clone();
                                                            
                                                            tokio::spawn(async move {
                                                                // --- TICK-TO-TRADE LATENCY LOGIC ---
                                                                let pre_trade_ts = std::time::SystemTime::now()
                                                                    .duration_since(std::time::UNIX_EPOCH)
                                                                    .unwrap_or_default()
                                                                    .as_micros();
                                                                
                                                                match client_clone.place_order(&base_payload, None, ob_context).await {
                                                                    Ok(response) => {
                                                                        let post_trade_ts = std::time::SystemTime::now()
                                                                            .duration_since(std::time::UNIX_EPOCH)
                                                                            .unwrap_or_default()
                                                                            .as_micros();
                                                                        println!("[LATENCY] Tick-to-Trade: {} us", post_trade_ts - arrival_time);
                                                                        println!("[M{}] SUCCESS: Order ID {}", m_id, response.order_id);
                                                                    },
                                                                    Err(e) => println!("[M{}] ERROR: {}", m_id, e),
                                                                }
                                                            });
                                                            
                                                        } else if model.state == ModelState::Shadow {
                                                            // SHADOW MODE (Virtual Tracking)
                                                            // (Logic remains the same as previous implementation)
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            Some(fill) = rx.recv() => {
                // --- INVENTORY UPDATE LOGIC ---
                println!("[INVENTORY] Received fill: {:?}", fill);
                
                // Extract model ID from orderLinkId
                if let Some(model_id_str) = fill.order_link_id.split('_').nth(1) {
                    if let Some(model_id) = model_id_str.strip_prefix('M').and_then(|id| id.parse::<u64>().ok()) {
                        
                        let mut state_guard = shared_state.lock().await;
                        if let Some(model) = state_guard.models.get_mut(&model_id) {
                            
                            // This is a simplified logic assuming one fill message per order.
                            // A robust implementation would track seen order_ids and calculate deltas.
                            if fill.order_status == "Filled" || fill.order_status == "PartiallyFilled" {
                                if let (Ok(qty), Ok(value)) = (Decimal::from_str(&fill.cum_exec_qty), Decimal::from_str(&fill.cum_exec_value)) {
                                    
                                    // This logic needs to be idempotent. We should only process a fill for a given order once.
                                    // For now, we assume one final "Filled" message.
                                    let cash_change = if fill.side == "Buy" { -value } else { value };
                                    let qty_change = if fill.side == "Buy" { qty } else { -qty };

                                    model.capital += cash_change;
                                    model.inventory += qty_change;
                                    
                                    // Only track realized PnL and loss for Live models
                                    if model.state == ModelState::Live {
                                        model.realized_pnl += cash_change;
                                        // Loss is the negative of PnL. We add the negative of the cash change.
                                        model.cumulative_loss += -cash_change.to_f64().unwrap_or(0.0);
                                    }

                                    println!("[INVENTORY] M{} updated. Inv: {}, Capital: {:.2}, PnL_Track: {:.4}", model.id, model.inventory, model.capital, model.realized_pnl);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}