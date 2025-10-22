use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;
use serde::Deserialize;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use std::sync::Arc;
use tokio::sync::Mutex;
use rust_decimal::Decimal;
use std::str::FromStr;

const BYBIT_PRIVATE_WS_URL: &str = "wss://stream-testnet.bybit.com/v5/private";

// --- Data Structures for Private Stream ---

// We use string deserialization for Decimals to ensure precision.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionData {
    pub symbol: String,
    pub exec_price: String,
    pub exec_qty: String,
    pub exec_fee: String,
    pub side: String,
    // CRITICAL: Used to link fills back to specific models
    pub order_link_id: String, 
}

#[derive(Debug, Deserialize)]
pub struct ExecutionMessage {
    pub topic: String,
    pub data: Vec<ExecutionData>,
}

// Define a trait that the GlobalState in main.rs must implement
pub trait InventoryManager: Send + Sync {
    fn update_inventory(&mut self, model_id: u64, qty_change: Decimal, cash_change: Decimal);
}

// Helper function for authentication
fn generate_signature(api_secret: &str, payload: &str) -> String {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes()).expect("HMAC key init failed");
    mac.update(payload.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Starts the private WebSocket stream handler task.
pub async fn start_private_stream<T: InventoryManager + 'static>(
    api_key: String,
    api_secret: String,
    state: Arc<Mutex<T>>,
) -> Result<()> {
    println!("[PRIVATE WS] Connecting...");
    let (ws_stream, _) = connect_async(BYBIT_PRIVATE_WS_URL).await?;
    let (mut write, mut read) = ws_stream.split();

    // --- 1. Authentication ---
    let expires = (chrono::Utc::now().timestamp_millis() + 5000).to_string();
    let payload = format!("GET/realtime{}", expires);
    let signature = generate_signature(&api_secret, &payload);

    let auth_msg = serde_json::json!({
        "op": "auth",
        "args": [api_key, expires, signature]
    });
    write.send(Message::Text(auth_msg.to_string())).await?;

    // --- 2. Subscription ---
    let sub_msg = serde_json::json!({
        "op": "subscribe",
        "args": ["execution"]
    });
    write.send(Message::Text(sub_msg.to_string())).await?;
    println!("[PRIVATE WS] Auth request sent and subscribed to 'execution'.");

    // --- 3. Event Loop and Ping/Pong ---
    // Bybit requires pings every 20-30 seconds.
    let mut ping_interval = tokio::time::interval(tokio::time::Duration::from_secs(20));

    loop {
        tokio::select! {
            // Send Ping every 20 seconds
            _ = ping_interval.tick() => {
                let ping_msg = serde_json::json!({"op": "ping"}).to_string();
                if write.send(Message::Text(ping_msg)).await.is_err() {
                    eprintln!("[PRIVATE WS] Ping failed. Connection likely lost.");
                    break;
                }
            },
            // Process incoming messages
            Some(msg) = read.next() => {
                 match msg {
                    Ok(Message::Text(text)) => {
                        // Handle Pong/Status messages
                        if text.contains(r#""op":"pong""#) || text.contains(r#""success":true"#) {
                            continue;
                        }

                        // Handle Execution Messages
                        if let Ok(exec_msg) = serde_json::from_str::<ExecutionMessage>(&text) {
                            if exec_msg.topic.starts_with("execution") {
                                let mut state_guard = state.lock().await;
                                for exec in exec_msg.data {
                                    process_execution(&mut *state_guard, exec);
                                }
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("[PRIVATE WS] Error: {}. Reconnection needed.", e);
                        break; // Exit loop on error
                    },
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn process_execution<T: InventoryManager>(state: &mut T, exec: ExecutionData) {
    // CRITICAL: Extract Model ID from orderLinkId (format: "asn_M{id}_{uuid}")
    let model_id = exec.order_link_id.split('_').nth(1)
        .and_then(|m_part| m_part.strip_prefix('M').and_then(|id_str| id_str.parse::<u64>().ok()));

    if model_id.is_none() {
        // Ignore fills not originating from the ASN system or liquidations (which don't need explicit tracking here)
        if !exec.order_link_id.starts_with("asn_LIQ") {
             return;
        }
    }
    
    // Handle both standard and liquidation fills
    let model_id = model_id.unwrap_or_else(|| {
        // Attempt to extract ID from liquidation linkID: "asn_LIQ_M{id}_{uuid}"
         exec.order_link_id.split('_').nth(2)
        .and_then(|m_part| m_part.strip_prefix('M').and_then(|id_str| id_str.parse::<u64>().ok()))
        .unwrap_or(0)
    });

    if model_id == 0 { return; }


    // Parse Decimals safely
    let price = Decimal::from_str(&exec.exec_price).unwrap_or_default();
    let qty = Decimal::from_str(&exec.exec_qty).unwrap_or_default();
    let fee = Decimal::from_str(&exec.exec_fee).unwrap_or_default();

    if price.is_zero() || qty.is_zero() { return; }

    let (qty_change, cash_change) = if exec.side == "Buy" {
        // Inventory increases, Cash decreases (value + fee)
        let value = price * qty;
        (qty, -(value + fee))
    } else {
        // Inventory decreases, Cash increases (value - fee)
        let value = price * qty;
        (-qty, value - fee)
    };

    // Update the specific model's inventory in GlobalState
    state.update_inventory(model_id, qty_change, cash_change);
}