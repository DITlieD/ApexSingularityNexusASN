document.addEventListener('DOMContentLoaded', (event) => {
    const connectionStatus = document.getElementById('connection-status');
    const latencyEl = document.getElementById('latency');
    const modelsBody = document.getElementById('models-body');
    const totalEquityEl = document.getElementById('total-equity');
    const activeModelsEl = document.getElementById('active-models');
    const totalPnlEl = document.getElementById('total-pnl');

    let socket;
    let equityHistory = [];
    const MAX_HISTORY_POINTS = 300; // Points on the chart (e.g., 60 seconds if updated 5 times/sec)

    // Initialize Chart (ApexCharts configuration)
    const chartOptions = {
        series: [{ name: 'Total Equity', data: [] }],
        chart: {
            id: 'realtime', height: 350, type: 'area',
            // Optimized for rapid updates
            animations: { enabled: true, easing: 'linear', dynamicAnimation: { speed: 200 } },
            toolbar: { show: false }, zoom: { enabled: false }, background: '#1e1e1e'
        },
        colors: ['#007acc'],
        dataLabels: { enabled: false },
        stroke: { curve: 'smooth', width: 2 },
        fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.7, opacityTo: 0.3, stops: [0, 100] } },
        xaxis: {
            type: 'datetime',
            labels: { style: { colors: '#aaaaaa' } }
        },
        yaxis: {
            labels: { 
                style: { colors: '#aaaaaa' },
                formatter: (value) => `$${value.toFixed(2)}`
            }
        },
        legend: { show: false },
        theme: { mode: 'dark' }
    };

    const chart = new ApexCharts(document.getElementById("equity-chart"), chartOptions);
    chart.render();


    function connect() {
        // Determine the WebSocket URL based on the current page location
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        socket = new WebSocket(wsUrl);

        socket.onopen = function(e) {
            console.log("[WS] Connection established");
            connectionStatus.textContent = "Connected";
            connectionStatus.className = "connected";
        };

        socket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                const latency = Date.now() - data.timestamp;
                latencyEl.textContent = `${latency}ms`;
                updateUI(data);
            } catch (e) {
                console.error("[WS] Error parsing message:", e);
            }
        };

        socket.onclose = function(event) {
            console.log('[WS] Connection closed. Reconnecting...');
            connectionStatus.textContent = "Disconnected";
            connectionStatus.className = "disconnected";
            latencyEl.textContent = 'N/A';
            setTimeout(connect, 2000); // Attempt to reconnect after 2 seconds
        };

        socket.onerror = function(error) {
            console.error("[WS] Error:", error);
            socket.close();
        };
    }

    function updateUI(data) {
        let totalEquity = 0;
        let activeModelsCount = 0;
        let totalPnl = 0;
        let rowsHtml = '';

        data.models.forEach(model => {
            totalEquity += model.equity;
            totalPnl += model.realized_pnl;
            // Check if the model state indicates activity
            if (model.state === 'Live' || model.state === 'Shadow') {
                activeModelsCount++;
            }

            // Format numbers for display
            const equityFormatted = model.equity.toFixed(4);
            const inventoryFormatted = model.inventory.toFixed(6);
            const pnlFormatted = model.realized_pnl.toFixed(4);
            const perfScoreFormatted = model.performance_score.toFixed(8);
            const pnlColor = model.realized_pnl >= 0 ? '#4caf50' : '#f44336';

            rowsHtml += `
                <tr>
                    <td>${model.id}</td>
                    <td>${model.symbol}</td>
                    <td class="state-${model.state}">${model.state}</td>
                    <td>$${equityFormatted}</td>
                    <td>${inventoryFormatted}</td>
                    <td style="color: ${pnlColor}">$${pnlFormatted}</td>
                    <td>${perfScoreFormatted}</td>
                    <td>${model.eval_ticks}</td>
                </tr>
            `;
        });

        // Efficiently update the DOM
        modelsBody.innerHTML = rowsHtml;
        totalEquityEl.textContent = `$${totalEquity.toFixed(2)}`;
        activeModelsEl.textContent = `${activeModelsCount} / ${data.models.length}`;
        totalPnlEl.textContent = `$${totalPnl.toFixed(4)}`;
        totalPnlEl.style.color = totalPnl >= 0 ? '#4caf50' : '#f44336';

        updateChart(data.timestamp, totalEquity);
    }

    function updateChart(timestamp, equity) {
        equityHistory.push([timestamp, equity]);
        if (equityHistory.length > MAX_HISTORY_POINTS) {
            equityHistory.shift(); // Keep the history size manageable
        }
        
        // Update the chart series data in real-time
        chart.updateSeries([{
            data: equityHistory
        }]);
    }

    connect();
});