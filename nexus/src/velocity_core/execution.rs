use anyhow::{Result, anyhow};
use hex;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use sha2::Sha256;
use std::sync::Arc;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use ort::{value::Value as OrtValue, session::Session, session::builder::GraphOptimizationLevel};
use ndarray::Array;

// --- Private WebSocket Module ---
pub mod private_ws {
    use anyhow::Result;
    use futures_util::{SinkExt, StreamExt};
    use serde::Deserialize;
    use serde_json::json;
    use tokio::sync::mpsc;
    use tokio_tungstenite::connect_async;
    use url::Url;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    const BYBIT_TESTNET_PRIVATE_WS_URL: &str = "wss://stream-testnet.bybit.com/v5/private";

    #[derive(Deserialize, Debug, Clone)]
    #[serde(rename_all = "camelCase")]
    pub struct PrivateOrderFill {
        pub symbol: String,
        pub order_id: String,
        pub order_link_id: String,
        pub side: String,
        pub cum_exec_qty: String,
        pub cum_exec_value: String,
        pub order_status: String,
    }

    #[derive(Deserialize, Debug)]
    struct PrivateWebsocketMessage {
        topic: String,
        data: Vec<PrivateOrderFill>,
    }

    pub async fn connect(
        api_key: &str,
        api_secret: &str,
        tx: mpsc::Sender<PrivateOrderFill>,
    ) -> Result<()> {
        let url = Url::parse(BYBIT_TESTNET_PRIVATE_WS_URL)?;
        let (mut ws_stream, _) = connect_async(url.as_str()).await?;
        println!("[PRIVATE STREAM] WebSocket connected.");

        let expires = (chrono::Utc::now() + chrono::Duration::seconds(10)).timestamp_millis();
        let signature_payload = format!("GET/realtime{}", expires);
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(api_secret.as_bytes())?;
        mac.update(signature_payload.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let auth_payload = json!({ "op": "auth", "args": [api_key, expires, signature] });
        ws_stream.send(tokio_tungstenite::tungstenite::Message::Text(auth_payload.to_string())).await?;
        
        let sub_payload = json!({ "op": "subscribe", "args": ["order"] });
        ws_stream.send(tokio_tungstenite::tungstenite::Message::Text(sub_payload.to_string())).await?;

        while let Some(msg) = ws_stream.next().await {
            match msg {
                Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                    if let Ok(parsed_msg) = serde_json::from_str::<PrivateWebsocketMessage>(&text) {
                        if parsed_msg.topic == "order" {
                            for fill in parsed_msg.data {
                                if tx.send(fill).await.is_err() {
                                    eprintln!("[PRIVATE STREAM] Failed to send fill on channel.");
                                }
                            }
                        }
                    } else if text.contains("auth success") {
                        println!("[PRIVATE STREAM] Authentication successful.");
                    } else if text.contains("subscribe") && text.contains("success") {
                         println!("[PRIVATE STREAM] Subscribed to 'order' topic.");
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("[PRIVATE STREAM] WebSocket error: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }
}

const BYBIT_TESTNET_API_URL: &str = "https://api-testnet.bybit.com";

// --- API Data Structures ---
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BybitTime { time_nano: String }
#[derive(Deserialize)]
struct TimeResult { result: BybitTime }

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct OrderResponse {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: OrderResult,
}
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct OrderResult {
    pub order_id: String,
    pub order_link_id: String,
}

// --- Execution Client ---
pub struct ExecutionClient {
    api_key: String,
    api_secret: String,
    client: Client,
    iel_session: Option<Arc<Session>>,
}

impl ExecutionClient {
    pub fn new(api_key: String, api_secret: String) -> Self {
        ExecutionClient {
            api_key,
            api_secret,
            client: Client::new(),
            iel_session: None,
        }
    }

    pub fn load_iel_model(&mut self, path: &str) -> Result<()> {
        println!("[IEL] Loading ONNX session from: {}", path);
        let session = Session::builder()?            .with_optimization_level(GraphOptimizationLevel::Level3)?            .with_intra_threads(1)?            .commit_from_file(path)?;        self.iel_session = Some(Arc::new(session));        println!("[IEL] Model loaded successfully.");        Ok(())    }
        
            async fn get_server_time(&self) -> Result<String> {
                let res = self.client.get(format!("{}/v5/market/time", BYBIT_TESTNET_API_URL)).send().await?.json::<TimeResult>().await?;
                Ok(res.result.time_nano[..res.result.time_nano.len() - 6].to_string())
            }
        
            pub async fn place_order(&self, base_payload: &Value, iel_state: Option<&Vec<Decimal>>, order_book_context: Option<(Decimal, Decimal)>) -> Result<OrderResult> {
                let mut payload = base_payload.clone();
                let tactic = self.determine_tactic(iel_state);
                self.apply_tactic(&mut payload, tactic, order_book_context)?;
        
                let timestamp = self.get_server_time().await?;
                let recv_window = "20000";
                let request_body = serde_json::to_string(&payload)?;
        
                let mut sig_payload = String::new();
                sig_payload.push_str(&timestamp);
                sig_payload.push_str(&self.api_key);
                sig_payload.push_str(recv_window);
                sig_payload.push_str(&request_body);
        
                type HmacSha256 = Hmac<Sha256>;
                let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(self.api_secret.as_bytes())?;
                mac.update(sig_payload.as_bytes());
                let signature = hex::encode(mac.finalize().into_bytes());
        
                let res = self.client.post(format!("{}/v5/order/create", BYBIT_TESTNET_API_URL))
                    .header("X-BAPI-API-KEY", &self.api_key)
                    .header("X-BAPI-TIMESTAMP", &timestamp)
                    .header("X-BAPI-SIGN", &signature)
                    .header("X-BAPI-RECV-WINDOW", recv_window)
                    .header("Content-Type", "application/json")
                    .body(request_body)
                    .send().await?;
        
                let res_text = res.text().await?;
                let order_response: OrderResponse = serde_json::from_str(&res_text)?;
        
                if order_response.ret_code == 0 {
                    Ok(order_response.result)
                } else {
                    Err(anyhow!("Bybit API Error: {} (ret_code: {})", order_response.ret_msg, order_response.ret_code))
                }
            }
        
            fn determine_tactic(&self, iel_state: Option<&Vec<Decimal>>) -> u32 {
                const DEFAULT_TACTIC: u32 = 0; // Aggressive Market Order
                if let (Some(session), Some(state)) = (&self.iel_session, iel_state) {
                    if state.len() != 4 { return DEFAULT_TACTIC; }
        
                    let state_f32: Vec<f32> = state.iter().map(|d| d.to_f32().unwrap_or(0.0)).collect();
                                if let Ok(input_tensor) = Array::from_shape_vec((1, 4), state_f32) {
                                    let inputs = [OrtValue::from_array(input_tensor.into_dyn()).unwrap()];
                                    if let Ok(outputs) = session.run(&inputs) {
                                        if let Ok(logits) = outputs[0].try_extract_tensor::<f32>() {
                                            let view = logits.view();
                                            return view.iter().enumerate()
                                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                                .map(|(i, _)| i as u32)
                                                .unwrap_or(DEFAULT_TACTIC);
                                        }
                                    }
                                }                }
                DEFAULT_TACTIC
    }

    fn apply_tactic(&self, payload: &mut Value, tactic: u32, context: Option<(Decimal, Decimal)>) -> Result<()> {
        let side_str = payload.get("side").and_then(|s| s.as_str()).ok_or_else(|| anyhow!("Invalid payload structure"))?;
        let side = side_str.to_string();
        
        match tactic {
            0 => { // Aggressive Market Order
                payload["orderType"] = Value::String("Market".to_string());
            },
            1 | 2 => { // Passive Limit Orders
                let (best_bid, best_ask) = context.ok_or_else(|| anyhow!("Missing order book context for passive order"))?;
                payload["orderType"] = Value::String("Limit".to_string());
                let tick_size = Decimal::new(1, 2);
                let price = if tactic == 1 { // At the touch
                    if side == "Buy" { best_bid } else { best_ask }
                } else { // Deep passive
                    if side == "Buy" { best_bid - tick_size } else { best_ask + tick_size }
                };

                if price > Decimal::ZERO {
                    payload["price"] = Value::String(price.to_string());
                } else {
                    payload["orderType"] = Value::String("Market".to_string()); // Fallback
                }
            },
            _ => { // Fallback
                payload["orderType"] = Value::String("Market".to_string());
            }
        }
        Ok(())
    }
}
