use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{BTreeMap, VecDeque}; // Import VecDeque
use std::str::FromStr;
use rust_decimal::prelude::ToPrimitive; // Import ToPrimitive

// NEW: Struct to hold historical snapshots for MTFA
#[derive(Clone, Copy)]
struct HistoricalSnapshot {
    mid_price: Decimal,
}

// --- Order Book Logic ---

pub struct OrderBook {
    pub symbol: String,
    pub bids: BTreeMap<std::cmp::Reverse<Decimal>, Decimal>,
    pub asks: BTreeMap<Decimal, Decimal>,
    pub last_update_id: u64,
    // NEW: History for MTFA
    history: VecDeque<HistoricalSnapshot>,
    max_history_len: usize,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        // Assuming roughly 1 update per second. 15 minutes = 900 seconds.
        let history_len = 900; 
        OrderBook {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            history: VecDeque::with_capacity(history_len + 1),
            max_history_len: history_len,
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

    pub fn get_mid_price(&self) -> Option<Decimal> {
        let best_bid = self.bids.keys().next().map(|r| r.0);
        let best_ask = self.asks.keys().next();
        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => Some((bid + *ask) / dec!(2.0)),
            _ => None,
        }
    }

    // NEW: Update historical data for MTFA
    fn update_history(&mut self, mid_price: Decimal) {
        self.history.push_back(HistoricalSnapshot { mid_price });
        if self.history.len() > self.max_history_len {
            self.history.pop_front();
        }
    }

    // NEW: Calculate MTFA features
    fn calculate_mtfa_features(&mut self) -> (Decimal, Decimal) {
        if self.history.is_empty() {
            return (Decimal::ZERO, Decimal::ZERO);
        }
        let mut returns = Vec::new();
        for window in self.history.make_contiguous().windows(2) {
            let p1 = window[0].mid_price;
            let p2 = window[1].mid_price;
            if p1 > Decimal::ZERO {
                returns.push((p2 - p1) / p1);
            }
        }

        if returns.len() < 2 {
            return (Decimal::ZERO, Decimal::ZERO);
        }

        let n = Decimal::from(returns.len());
        let mean = returns.iter().sum::<Decimal>() / n;
        let var = returns.iter().map(|r| (*r - mean) * (*r - mean)).sum::<Decimal>() / (n - dec!(1));
        let volatility = var.to_f64().and_then(|v| Decimal::from_f64_retain(v.sqrt())).unwrap_or(dec!(0));

        (mean, volatility)
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

    // MODIFIED: Integrate MTFA and history updates
    pub fn calculate_microstructure_features(&mut self) -> Option<MicrostructureFeatures> {
        
        if let (Some(best_bid), Some(best_ask)) = (self.bids.iter().next(), self.asks.iter().next()) {
            let bid_size = *best_bid.1;
            let ask_size = *best_ask.1;
            let total_top_volume = bid_size + ask_size;

            let imbalance = if total_top_volume > dec!(0) {
                (bid_size - ask_size) / total_top_volume
            } else {
                dec!(0)
            };
            
            // (Book pressure calculation omitted for brevity - use existing implementation)
            let pressure = imbalance; // Placeholder

            let mid_price = (best_bid.0.0 + *best_ask.0) / dec!(2.0);

            // Update History
            self.update_history(mid_price);

            // Calculate MTFA
            let (vol_15m, mom_15m) = self.calculate_mtfa_features();

            Some(MicrostructureFeatures {
                top_of_book_imbalance: imbalance,
                book_pressure: pressure,
                clv_proxy: dec!(0.5),
                // NEW MTFA Features
                volatility_15m: vol_15m,
                momentum_15m: mom_15m,
            })
        } else {
            None
        }
    }

    pub fn get_spread(&self) -> Option<Decimal> {
        let best_bid = self.bids.keys().next().map(|r| r.0);
        let best_ask = self.asks.keys().next();
        if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
            Some(*ask - bid)
        } else {
            None
        }
    }

    // NEW: Get the Best Bid/Ask (BBO) context for passive orders
    pub fn get_bbo(&self) -> (Option<Decimal>, Option<Decimal>) {
         let best_bid = self.bids.keys().next().map(|r| r.0);
         let best_ask = self.asks.keys().next().cloned();
         (best_bid, best_ask)
    }
    
    // NEW: Calculate IEL State features (Must match Python definition: STATE_DIM=4)
    // State: [Imbalance, Spread (normalized), Pressure, Signal Strength]
    pub fn calculate_iel_state(&self, micro_features: &MicrostructureFeatures, signal_strength: Decimal) -> Option<Vec<Decimal>> {
        
        if let Some(spread) = self.get_spread() {
            // Normalize spread. We use 50.0 as a baseline normalization factor. 
            // This factor should be calibrated based on the market (e.g., BTC vs SOL).
            let max_spread = dec!(50.0);
            let normalized_spread = (spread / max_spread).clamp(dec!(0), dec!(1));

            Some(vec![
                micro_features.top_of_book_imbalance,
                normalized_spread,
                micro_features.book_pressure,
                // Normalize signal strength (-1 to 1)
                signal_strength.clamp(dec!(-1), dec!(1)), 
            ])
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct MicrostructureFeatures {
    pub top_of_book_imbalance: Decimal,
    pub book_pressure: Decimal,
    pub clv_proxy: Decimal,
    // MTFA Features
    pub volatility_15m: Decimal,
    pub momentum_15m: Decimal,
}