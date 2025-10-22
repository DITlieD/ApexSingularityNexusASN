use axum::{
    routing::get,
    Router,
    extract::{ws::{WebSocketUpgrade, WebSocket, Message}, State},
};
use serde::Serialize;
use tokio::sync::broadcast;
use tower_http::services::ServeDir;
use futures_util::{sink::SinkExt, stream::StreamExt};
use std::net::SocketAddr;

// Define the data structure for the UI (Uses f64 for easy serialization/charting)
#[derive(Serialize, Debug, Clone)]
pub struct UiModelState {
    pub id: u64,
    pub symbol: String,
    pub state: String,
    pub equity: f64,
    pub inventory: f64,
    pub realized_pnl: f64,
    pub performance_score: f64,
    pub eval_ticks: u64,
}

#[derive(Serialize, Debug, Clone)]
pub struct UiBroadcastState {
    pub timestamp: i64,
    pub models: Vec<UiModelState>,
}

// Application State for Axum
#[derive(Clone)]
struct AppState {
    pub tx: broadcast::Sender<UiBroadcastState>,
}

pub async fn start_ui_server(tx: broadcast::Sender<UiBroadcastState>) {
    let app_state = AppState { tx };

    // Determine the path to the UI static files
    // We check relative paths robustly.
    let base_path = std::env::current_dir().unwrap_or_default();
    let ui_path_root = base_path.join("nexus/ui/dist");
    let ui_path_local = base_path.join("ui/dist");

    let ui_path = if ui_path_root.exists() {
        ui_path_root
    } else if ui_path_local.exists() {
        ui_path_local
    } else {
        eprintln!("FATAL: UI directory 'nexus/ui/dist' or 'ui/dist' not found. Ensure the UI files exist.");
        return;
    };

    // Configure static file serving
    let serve_dir = ServeDir::new(ui_path)
        .append_index_html(true)
        .precompressed_gzip();

    let app = Router::new()
        // WebSocket endpoint
        .route("/ws", get(ws_handler))
        // Serve the UI files
        .fallback_service(serve_dir)
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3030));
    
    // Attempt to bind the listener
    match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => {
            println!("[UI SERVER] Dashboard running at http://localhost:3030");
            axum::serve(listener, app).await.unwrap();
        },
        Err(e) => {
            eprintln!("FATAL: Could not bind UI server to {}: {}", addr, e);
        }
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    // Subscribe to the broadcast channel
    let mut rx = state.tx.subscribe();

    // Task to forward broadcast messages to the WebSocket client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            // Serialize the state update
            let json_msg = serde_json::to_string(&msg).unwrap_or_default();
            if sender.send(Message::Text(json_msg)).await.is_err() {
                // Client disconnected
                break;
            }
        }
    });

    // Task to handle incoming messages (e.g., pings or close events)
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if matches!(msg, Message::Close(_)) {
                break;
            }
        }
    });

    // If either task finishes, abort the other
    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    };
}