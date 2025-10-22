use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{BTreeMap, HashMap, BinaryHeap};
use std::cmp::Reverse;
use crate::inference::gp_interpreter::{GPStrategy, HybridStrategy, StrategyOutput};

use rand_distr::{LogNormal, Distribution};
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};

// (Event-Driven Simulation Components: SimEvent, EventType - unchanged)
#[derive(Debug)]
struct SimEvent { timestamp: u128, event_type: EventType }
impl PartialEq for SimEvent { fn eq(&self, other: &Self) -> bool { self.timestamp == other.timestamp } }
impl Eq for SimEvent {}
impl PartialOrd for SimEvent { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl Ord for SimEvent { fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.timestamp.cmp(&other.timestamp) } }
#[derive(Debug)]
enum EventType { AgentDecision { agent_id: u64 }, OrderArrival { order: Order } }

// (Configuration: SimulationConfig - unchanged)
#[derive(Debug, Clone)]
pub struct SimulationConfig { pub num_ticks: u64, pub taker_fee_rate: Decimal, pub maker_fee_rate: Decimal, pub initial_capital: Decimal, pub target_capital_factor: Decimal, pub slippage_factor: Decimal }

// (Core Data Structures: Order, Trade, AgentState, Agent trait - unchanged)
#[derive(Debug, Clone, Copy)]
pub struct Order { pub agent_id: u64, pub price: Decimal, pub size: Decimal, pub is_bid: bool }
#[derive(Debug, Clone, Copy)]
pub struct Trade { pub aggressor_agent_id: u64, pub resting_agent_id: u64, pub price: Decimal, pub size: Decimal, pub aggressor_fee: Decimal, pub resting_fee: Decimal, pub aggressor_was_bid: bool }
#[derive(Debug, Clone)]
pub struct AgentState { pub cash: Decimal, pub inventory: Decimal }
impl AgentState {
    fn new(initial_capital: Decimal) -> Self { Self { cash: initial_capital, inventory: Decimal::ZERO } }
    fn calculate_equity(&self, mid_price: Option<Decimal>) -> Decimal { self.cash + self.inventory * mid_price.unwrap_or(Decimal::ZERO) }
}
pub trait Agent: Send { fn on_tick(&mut self, market_state: &MarketState, agent_state: &AgentState) -> Vec<Order>; fn id(&self) -> u64; }

// (MarketState - with full implementation)
#[derive(Debug, Clone)]
pub struct MarketState { pub bids: BTreeMap<Reverse<Decimal>, Vec<Order>>, pub asks: BTreeMap<Decimal, Vec<Order>>, pub sequence: u64 }
impl MarketState {
    fn new() -> Self { Self { bids: BTreeMap::new(), asks: BTreeMap::new(), sequence: 0 } }
    pub fn get_mid_price(&self) -> Option<Decimal> {
        let best_bid = self.bids.keys().next().map(|r| r.0);
        let best_ask = self.asks.keys().next();
        if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
            Some((bid + *ask) / dec!(2.0))
        } else {
            None
        }
    }
    pub fn calculate_features(&self) -> Option<Vec<Decimal>> {
        // (Full implementation restored)
        let best_bid = self.bids.iter().next();
        let best_ask = self.asks.iter().next();
        if let (Some((_, bid_orders)), Some((_, ask_orders))) = (best_bid, best_ask) {
            let bid_size: Decimal = bid_orders.iter().map(|o| o.size).sum();
            let ask_size: Decimal = ask_orders.iter().map(|o| o.size).sum();
            let total_top_volume = bid_size + ask_size;
            let imbalance = if total_top_volume > dec!(0) { (bid_size - ask_size) / total_top_volume } else { dec!(0) };
            let bid_volume_top5: Decimal = self.bids.iter().take(5).flat_map(|(_, orders)| orders.iter().map(|o| o.size)).sum();
            let ask_volume_top5: Decimal = self.asks.iter().take(5).flat_map(|(_, orders)| orders.iter().map(|o| o.size)).sum();
            let total_volume_top5 = bid_volume_top5 + ask_volume_top5;
            let pressure = if total_volume_top5 > dec!(0) { (bid_volume_top5 - ask_volume_top5) / total_volume_top5 } else { dec!(0) };
            return Some(vec![imbalance, pressure, dec!(0.5)]);
        }
        None
    }
}

// (MarketStatistics - unchanged)
#[derive(Debug, Clone, Default)]
pub struct MarketStatistics { pub total_volume: Decimal, pub realized_volatility: Decimal, pub avg_spread: Decimal, pub trade_count: u64, spread_sum: Decimal, mid_price_history: Vec<Decimal> }

// --- Matching Engine (Full implementation restored) ---
pub struct MatchingEngine;
impl MatchingEngine {
    pub fn process_order(market_state: &mut MarketState, mut order: Order, config: &SimulationConfig) -> Vec<Trade> {
        let mut trades = Vec::new();
        if order.is_bid { Self::match_bid(market_state, &mut order, &mut trades, config); } 
        else { Self::match_ask(market_state, &mut order, &mut trades, config); }
        if order.size > Decimal::ZERO { Self::add_order_to_book(market_state, order); }
        trades
    }
    fn match_bid(market_state: &mut MarketState, aggressing_order: &mut Order, trades: &mut Vec<Trade>, config: &SimulationConfig) {
        let mut filled_levels = Vec::new();
        for (ask_price, orders_at_level) in market_state.asks.iter_mut() {
            if aggressing_order.price < *ask_price { break; }
            for resting_order in orders_at_level.iter_mut() {
                let trade_size = aggressing_order.size.min(resting_order.size);
                let slippage = resting_order.price * config.slippage_factor * trade_size;
                let effective_trade_price = resting_order.price + slippage;
                let trade_value = trade_size * effective_trade_price;
                trades.push(Trade { aggressor_agent_id: aggressing_order.agent_id, resting_agent_id: resting_order.agent_id, price: effective_trade_price, size: trade_size, aggressor_fee: trade_value * config.taker_fee_rate, resting_fee: trade_value * config.maker_fee_rate, aggressor_was_bid: true });
                aggressing_order.size -= trade_size;
                resting_order.size -= trade_size;
                if aggressing_order.size == Decimal::ZERO { break; }
            }
            orders_at_level.retain(|o| o.size > Decimal::ZERO);
            if orders_at_level.is_empty() { filled_levels.push(*ask_price); }
            if aggressing_order.size == Decimal::ZERO { break; }
        }
        for price in filled_levels { market_state.asks.remove(&price); }
    }
    fn match_ask(market_state: &mut MarketState, aggressing_order: &mut Order, trades: &mut Vec<Trade>, config: &SimulationConfig) {
        let mut filled_levels = Vec::new();
        for (bid_price, orders_at_level) in market_state.bids.iter_mut() {
            if aggressing_order.price > bid_price.0 { break; }
            for resting_order in orders_at_level.iter_mut() {
                let trade_size = aggressing_order.size.min(resting_order.size);
                let slippage = resting_order.price * config.slippage_factor * trade_size;
                let effective_trade_price = resting_order.price - slippage;
                let trade_value = trade_size * effective_trade_price;
                trades.push(Trade { aggressor_agent_id: aggressing_order.agent_id, resting_agent_id: resting_order.agent_id, price: effective_trade_price, size: trade_size, aggressor_fee: trade_value * config.taker_fee_rate, resting_fee: trade_value * config.maker_fee_rate, aggressor_was_bid: false });
                aggressing_order.size -= trade_size;
                resting_order.size -= trade_size;
                if aggressing_order.size == Decimal::ZERO { break; }
            }
            orders_at_level.retain(|o| o.size > Decimal::ZERO);
            if orders_at_level.is_empty() { filled_levels.push(*bid_price); }
            if aggressing_order.size == Decimal::ZERO { break; }
        }
        for price in filled_levels { market_state.bids.remove(&price); }
    }
    fn add_order_to_book(market_state: &mut MarketState, order: Order) {
        if order.is_bid { market_state.bids.entry(Reverse(order.price)).or_default().push(order); } 
        else { market_state.asks.entry(order.price).or_default().push(order); }
    }
}

pub struct Simulation {
    config: SimulationConfig,
    market_state: MarketState,
    agents: HashMap<u64, Box<dyn Agent>>,
    agent_states: HashMap<u64, AgentState>,
    event_queue: BinaryHeap<Reverse<SimEvent>>,
    current_tick: u64,
    pub statistics: MarketStatistics,
}

#[derive(Debug)]
pub struct SimulationResult {
    pub final_equity: Decimal,
    pub ttt_fitness: f64,
}

impl Simulation {
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            config,
            market_state: MarketState::new(),
            agents: HashMap::new(),
            agent_states: HashMap::new(),
            event_queue: BinaryHeap::new(),
            current_tick: 0,
            statistics: MarketStatistics::default(),
        }
    }

    pub fn add_agent(&mut self, mut agent: Box<dyn Agent>) {
        let agent_id = agent.id();
        self.agent_states.insert(agent_id, AgentState::new(self.config.initial_capital));
        self.agents.insert(agent_id, agent);
    }

    fn schedule_event(&mut self, event: SimEvent) {
        self.event_queue.push(Reverse(event));
    }

    fn get_latency(&self) -> u128 {
        // TODO: Implement a more realistic latency model
        1_000_000 // 1ms in nanoseconds
    }

    pub fn run(&mut self, ttt_agent_id: u64) -> SimulationResult {
                let agent_ids: Vec<u64> = self.agents.keys().cloned().collect();
                for agent_id in agent_ids {
                    self.schedule_event(SimEvent {
                        timestamp: 0,
                        event_type: EventType::AgentDecision { agent_id },
                    });
                }
        while let Some(Reverse(event)) = self.event_queue.pop() {
            if self.current_tick >= self.config.num_ticks {
                break;
            }
            self.process_event(event);
        }

        let final_equity = self.agent_states.get(&ttt_agent_id).unwrap().calculate_equity(self.market_state.get_mid_price());
        let target_equity = self.config.initial_capital * self.config.target_capital_factor;
        let ttt_fitness = if final_equity >= target_equity {
            (self.config.num_ticks - self.current_tick) as f64
        } else {
            -((target_equity - final_equity) / target_equity).to_f64().unwrap_or(1.0) * (self.config.num_ticks as f64)
        };

        SimulationResult {
            final_equity,
            ttt_fitness,
        }
    }

    fn process_event(&mut self, event: SimEvent) {
        self.current_tick = (event.timestamp / 1_000_000_000) as u64; // Convert ns to s for tick
        match event.event_type {
            EventType::AgentDecision { agent_id } => {
                if let (Some(agent), Some(agent_state)) = (self.agents.get_mut(&agent_id), self.agent_states.get(&agent_id)) {
                    let orders = agent.on_tick(&self.market_state, agent_state);
                    for order in orders {
                        self.schedule_event(SimEvent {
                            timestamp: event.timestamp + self.get_latency(),
                            event_type: EventType::OrderArrival { order },
                        });
                    }
                }
                // Schedule next decision
                self.schedule_event(SimEvent {
                    timestamp: event.timestamp + 1_000_000_000, // 1 second in ns
                    event_type: EventType::AgentDecision { agent_id },
                });
            }
            EventType::OrderArrival { order } => {
                let trades = MatchingEngine::process_order(&mut self.market_state, order, &self.config);
                for trade in trades {
                    self.update_agent_state(&trade);
                    self.update_statistics(&trade);
                }
            }
        }
    }
    fn update_statistics(&mut self, trade: &Trade) {
        self.statistics.total_volume += trade.size;
        self.statistics.trade_count += 1;
        if let Some(mid_price) = self.market_state.get_mid_price() {
            self.statistics.mid_price_history.push(mid_price);
            if self.statistics.mid_price_history.len() > 100 {
                self.statistics.mid_price_history.remove(0);
            }
            let best_bid = self.market_state.bids.keys().next().map(|r| r.0);
            let best_ask = self.market_state.asks.keys().next();
            if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
                let spread = *ask - bid;
                self.statistics.spread_sum += spread;
                self.statistics.avg_spread = self.statistics.spread_sum / Decimal::from(self.statistics.trade_count);
            }
        }
    }
    fn update_agent_state(&mut self, trade: &Trade) {
        let trade_value = trade.size * trade.price;
        if let Some(state) = self.agent_states.get_mut(&trade.aggressor_agent_id) {
            if trade.aggressor_was_bid {
                state.inventory += trade.size;
                state.cash -= trade_value;
            } else {
                state.inventory -= trade.size;
                state.cash += trade_value;
            }
            state.cash -= trade.aggressor_fee;
        }
        if let Some(state) = self.agent_states.get_mut(&trade.resting_agent_id) {
            if trade.aggressor_was_bid {
                state.inventory -= trade.size;
                state.cash += trade_value;
            } else {
                state.inventory += trade.size;
                state.cash -= trade_value;
            }
            state.cash -= trade.resting_fee;
        }
    }
}
// --- Agent Implementations (Full, updated logic) ---
const APEX_PREDATOR_ID: u64 = 1;
const DSG_AGENT_ID: u64 = 2;

struct ApexPredatorAgent { strategy: HybridStrategy }
impl Agent for ApexPredatorAgent {
    fn id(&self) -> u64 { APEX_PREDATOR_ID }
    fn on_tick(&mut self, market_state: &MarketState, agent_state: &AgentState) -> Vec<Order> {
        if let Some(features) = market_state.calculate_features() {
            if let Ok(output) = self.strategy.evaluate(&features, None) {
                let p = output.win_probability;
                let q = dec!(1.0) - p;
                let b = dec!(3.0);
                let optimal_f = (b * p - q) / b;
                if optimal_f > dec!(0.0) {
                    let trade_fraction = dec!(0.5) * optimal_f;
                    let trade_size_dollars = agent_state.cash * trade_fraction;
                    if let Some(price) = market_state.get_mid_price() {
                        if price > dec!(0) {
                            let qty = trade_size_dollars / price;
                            if p > dec!(0.5) { // Buy
                                if let Some(best_ask) = market_state.asks.keys().next() {
                                    return vec![Order { agent_id: self.id(), price: *best_ask, size: qty, is_bid: true }];
                                }
                            } else { // Sell
                                if let Some(best_bid) = market_state.bids.keys().next() {
                                    return vec![Order { agent_id: self.id(), price: best_bid.0, size: qty, is_bid: false }];
                                }
                            }
                        }
                    }
                }
            }
        }
        vec![]
    }
}

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

struct DSGAgent { genome: Vec<f64>, rng: StdRng }
impl DSGAgent { fn new(genome: Vec<f64>) -> Self { Self { genome, rng: StdRng::from_entropy() } } }
impl Agent for DSGAgent {
    fn id(&self) -> u64 { DSG_AGENT_ID }
    fn on_tick(&mut self, market_state: &MarketState, agent_state: &AgentState) -> Vec<Order> {
        let order_prob = self.genome.get(0).cloned().unwrap_or(0.8);
        if self.rng.gen_bool(order_prob) {
            let spread_factor = self.genome.get(1).cloned().unwrap_or(0.01);
            let inventory_sensitivity = self.genome.get(2).cloned().unwrap_or(0.0001);
            let order_size_param = self.genome.get(3).cloned().unwrap_or(1.0);
            let reference_price = market_state.get_mid_price().unwrap_or(dec!(10000.0));
            let sensitivity = Decimal::from_f64(inventory_sensitivity).unwrap_or_default();
            let inventory_skew = agent_state.inventory * sensitivity * reference_price;
            let adjusted_reference_price = reference_price - inventory_skew;
            if adjusted_reference_price <= dec!(0.01) { return vec![]; }
            let spread = adjusted_reference_price * Decimal::from_f64(spread_factor).unwrap_or_default();
            if spread <= dec!(0) { return vec![]; }
            let buy_price = (adjusted_reference_price - spread / dec!(2)).round_dp(2);
            let sell_price = (adjusted_reference_price + spread / dec!(2)).round_dp(2);
            let size = Decimal::from_f64(order_size_param).unwrap_or_default();
            let mut orders = Vec::new();
            if buy_price > dec!(0) { orders.push(Order { agent_id: self.id(), price: buy_price, size, is_bid: true }); }
            if sell_price > buy_price { orders.push(Order { agent_id: self.id(), price: sell_price, size, is_bid: false }); }
            return orders;
        }
        vec![]
    }
}

// --- Accelerated Simulation Runner (Full, updated logic) ---
pub fn run_accelerated_simulation(config: SimulationConfig, apex_strategy_json: String, apex_onnx_path: Option<String>, dsg_parameters: Vec<f64>) -> SimulationResult {
    let mut sim = Simulation::new(config.clone());
    let apex_strategy = match HybridStrategy::new(&apex_strategy_json, apex_onnx_path) {
        Ok(s) => s,
        Err(_) => return SimulationResult { final_equity: dec!(0), ttt_fitness: -(config.num_ticks as f64) * 10.0 },
    };
    sim.add_agent(Box::new(ApexPredatorAgent { strategy: apex_strategy }));
    sim.add_agent(Box::new(DSGAgent::new(dsg_parameters)));
    sim.run(APEX_PREDATOR_ID)
}
// (run_accelerated_chimera_simulation needs to be added back)
pub fn run_accelerated_chimera_simulation(config: SimulationConfig, dsg_parameters: Vec<f64>) -> MarketStatistics {
    let mut sim = Simulation::new(config);
    sim.add_agent(Box::new(DSGAgent::new(dsg_parameters)));
    // sim.run_chimera() // This method needs to be re-implemented
    sim.statistics
}
