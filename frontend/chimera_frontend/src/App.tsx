import React, { useState, useEffect, useCallback } from 'react';
import * as api from './api';
import { Environment, GraphData } from './api';
import NetworkVisualizer from './NetworkVisualizer';
import PerformanceChart from './PerformanceChart';
import './App.css';

interface TrainingMetric {
  episode: number;
  reward: number;
  avg_reward: number;
  epsilon: number;
  policy_loss: number;
}

function App() {
  const AGENT_ID = "Kymera-local-train"; // Hardcoded agent ID

  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [selectedEnv, setSelectedEnv] = useState<string>('');
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [textInput, setTextInput] = useState<string>('');
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [statusMessage, setStatusMessage] = useState('Welcome to Project Chimera!');

  const fetchEnvironments = useCallback(async () => {
    try {
      const response = await api.getEnvironments();
      setEnvironments(response.data);
      if (response.data.length > 0) {
        setSelectedEnv(response.data[0].id);
      }
    } catch (error) {
      console.error("Failed to fetch environments:", error);
      setStatusMessage('Error: Could not fetch environments.');
    }
  }, []);

  useEffect(() => {
    // Fetch initial data on component mount
    fetchEnvironments();

    setStatusMessage(`Fetching structure for ${AGENT_ID}...`);
    api.getAgentStructure(AGENT_ID)
      .then(response => {
        setGraphData(response.data);
        setStatusMessage(`Visualizing agent: ${AGENT_ID}`);
      })
      .catch(error => {
        console.error("Failed to fetch graph structure:", error);
        setStatusMessage('Failed to fetch graph structure.');
      });
  }, [fetchEnvironments, AGENT_ID]); // AGENT_ID is a constant, but including it makes dependencies explicit

  const handleLearn = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!textInput) return;
    try {
      setStatusMessage(`Agent is learning pattern: \"${textInput}\"...`);
      // CORRECTED: Use the updated learnWithAgent function.
      await api.learnWithAgent(AGENT_ID, textInput);
      setTextInput('');
      setStatusMessage('Learning complete. Refreshing structure...');

      const response = await api.getAgentStructure(AGENT_ID);
      setGraphData(response.data);
      setStatusMessage(`Visualizing agent: ${AGENT_ID}`);
    } catch (error) {
      console.error("Failed to learn pattern:", error);
      setStatusMessage('Failed to learn pattern.');
    }
  };

  const handleStartTraining = async () => {
    if (!selectedEnv) return;
    try {
      setStatusMessage(`Starting training for ${AGENT_ID} in ${selectedEnv}...`);
      await api.startTraining(AGENT_ID, selectedEnv);
      setStatusMessage(`Training started. Waiting for metrics...`);
      setTrainingMetrics([]); // Clear previous metrics
    } catch (error) {
      console.error("Failed to start training:", error);
      setStatusMessage('Failed to start training.');
    }
  };

  // Connect to the general training websocket when the app loads
  useEffect(() => {
    const ws_scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const ws_path = `${ws_scheme}://${window.location.host.replace('3000', '8000')}/ws/brain/`;

    const socket = new WebSocket(ws_path);

    socket.onopen = () => {
      console.log("WebSocket connected for general training metrics.");
      setStatusMessage("Connected to real-time training metrics.");
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received metric:", data);
      // The data from the backend is the metric itself
      setTrainingMetrics(prevMetrics => [...prevMetrics.slice(-9), data]); // Keep last 10 metrics
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected.");
      setStatusMessage("Disconnected from real-time training metrics.");
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatusMessage("Error connecting to real-time metrics.");
    };

    // Cleanup on component unmount
    return () => {
      socket.close();
    };
  }, []); // Empty dependency array ensures this runs only once


  const [activeTab, setActiveTab] = useState('performance');

  return (
    <div className="App">
      <header className="App-header">
        <h1>Project Chimera</h1>
      </header>
      <div className="main-content">
        <div className="left-panel">
            <div className="actions">
              <div className="control-group">
                <h4>Cognitive Learning (Agent: {AGENT_ID})</h4>
                <form onSubmit={handleLearn}>
                  <input
                    type="text"
                    value={textInput}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTextInput(e.target.value)}
                    placeholder="Enter a pattern to learn"
                  />
                  <button type="submit">Learn & Organize</button>
                </form>
              </div>
              <div className="control-group">
                <h4>Reinforcement Learning</h4>
                <select onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSelectedEnv(e.target.value)} value={selectedEnv}>
                  {environments.map(env => (
                    <option key={env.id} value={env.id}>{env.name}</option>
                  ))}
                </select>
                <button onClick={handleStartTraining} disabled={!selectedEnv}>
                  Start Training
                </button>
              </div>
            </div>
        </div>

        <main className="visualizer-container">
          <NetworkVisualizer graphData={graphData} />
          <div className="status-bar">
            {statusMessage}
          </div>
          <div className="metrics-container">
            {/* Tab buttons would go here */}
            <h3>Performance Dashboard</h3>
            <PerformanceChart data={trainingMetrics} />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
