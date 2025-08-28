import React, { useState, useEffect } from 'react';
import NetworkVisualizer, { GraphData } from './NetworkVisualizer';
import PerformanceChart from './PerformanceChart';
import './App.css';

// --- Type Definitions ---

interface Environment {
  id: string;
  name: string;
}

interface TrainingMetric {
  episode: number;
  reward: number;
  avg_reward: number;
  epsilon: number;
  policy_loss: number;
  total_steps: number;
}

const AGENT_ID = "Kymera-local-train";

function App() {
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [statusMessage, setStatusMessage] = useState('Connecting to backend...');

  useEffect(() => {
    const ws_scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const ws_path = `${ws_scheme}://${window.location.host.replace(':3001', ':8001')}/ws/brain/`;

    const socket = new WebSocket(ws_path);

    socket.onopen = () => {
      console.log("WebSocket connected.");
      setStatusMessage("Connected to backend. Waiting for data...");
    };

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const { type, payload } = message;

      switch (type) {
        case 'environments_update':
          setEnvironments(payload);
          setStatusMessage("Environment list received.");
          break;
        case 'graph_update':
          setGraphData(payload);
          setStatusMessage(`Graph structure updated for ${AGENT_ID}.`);
          break;
        case 'training_metrics':
          setTrainingMetrics(prevMetrics => {
            const newMetrics = [...prevMetrics, payload];
            // Keep the last 100 metrics for a smoother chart
            return newMetrics.slice(-100);
          });
          break;
        default:
          console.warn("Received unknown message type:", type);
      }
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected.");
      setStatusMessage("Disconnected from backend. Please refresh to reconnect.");
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatusMessage("WebSocket connection error.");
    };

    // Cleanup on component unmount
    return () => {
      socket.close();
    };
  }, []); // Empty dependency array ensures this runs only once

  return (
    <div className="App">
      <header className="App-header">
        <h1>Project Chimera: Real-time Monitor</h1>
      </header>
      <div className="main-content">
        <div className="left-panel">
          <div className="control-group">
            <h4>Agent</h4>
            <p>{AGENT_ID}</p>
          </div>
          <div className="control-group">
            <h4>Environments</h4>
            {environments.length > 0 ? (
              <ul>
                {environments.map(env => <li key={env.id}>{env.name}</li>)}
              </ul>
            ) : (
              <p>Waiting for environment data...</p>
            )}
          </div>
        </div>

        <main className="visualizer-container">
          <NetworkVisualizer graphData={graphData} />
          <div className="status-bar">
            {statusMessage}
          </div>
          <div className="metrics-container">
            <h3>Performance Dashboard</h3>
            <PerformanceChart data={trainingMetrics} />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
