import React, { useState, useEffect } from 'react';
import NetworkVisualizer, { GraphData } from './NetworkVisualizer';
import PerformanceChart from './PerformanceChart';
import ActionVisualizer from './ActionVisualizer';
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
  const [actionProbs, setActionProbs] = useState<number[]>([]);

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
          const newMetric: TrainingMetric = {
            episode: payload.episode ?? (trainingMetrics[trainingMetrics.length - 1]?.episode ?? 0),
            reward: payload.reward ?? 0,
            avg_reward: payload.avg_reward ?? 0,
            epsilon: payload.epsilon ?? 0,
            policy_loss: payload.policy_loss ?? 0,
            total_steps: payload.total_steps ?? 0,
          };
          setTrainingMetrics(prevMetrics => {
            const newMetrics = [...prevMetrics, newMetric];
            return newMetrics.slice(-100);
          });
          break;
        case 'action_update':
            setActionProbs(payload.probabilities);
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

    return () => {
      socket.close();
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Project Chimera: Real-time Monitor</h1>
        <div className="status-bar">
          {statusMessage}
        </div>
      </header>
      <div className="main-content-grid">
        <div className="top-row">
          <div className="network-visualizer-container">
            <h3>STAG Graph</h3>
            <NetworkVisualizer graphData={graphData} />
          </div>
          <div className="performance-chart-container">
            <h3>Performance Dashboard</h3>
            <PerformanceChart data={trainingMetrics} />
          </div>
        </div>
        <div className="bottom-row">
          <h3>Action Probabilities</h3>
          <ActionVisualizer probabilities={actionProbs} />
        </div>
      </div>
    </div>
  );
}

export default App;
