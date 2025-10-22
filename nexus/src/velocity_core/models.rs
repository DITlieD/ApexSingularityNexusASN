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
