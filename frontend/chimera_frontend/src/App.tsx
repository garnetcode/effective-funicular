import React, { useState, useEffect, useCallback } from 'react';
import * as api from './api';
import { Agent, Environment, GraphData } from './api';
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
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [selectedEnv, setSelectedEnv] = useState<string>('');
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [textInput, setTextInput] = useState<string>('');
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [statusMessage, setStatusMessage] = useState('Welcome to Project Chimera!');

  const fetchAgents = useCallback(async () => {
    try {
      const response = await api.getAgents();
      setAgents(response.data);
    } catch (error) {
      console.error("Failed to fetch agents:", error);
      setStatusMessage('Error: Could not connect to backend.');
    }
  }, []);

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
    fetchAgents();
    fetchEnvironments();
  }, [fetchAgents, fetchEnvironments]);

  useEffect(() => {
    if (selectedAgent) {
      setStatusMessage(`Fetching structure for ${selectedAgent}...`);
      api.getAgentStructure(selectedAgent)
        .then(response => {
          setGraphData(response.data);
          setStatusMessage(`Visualizing agent: ${selectedAgent}`);
        })
        .catch(error => {
          console.error("Failed to fetch graph structure:", error);
          setStatusMessage('Failed to fetch graph structure.');
        });
    } else {
      setGraphData(null);
    }
  }, [selectedAgent]);

  const handleCreateAgent = async () => {
    try {
      setStatusMessage('Creating new agent...');
      // CORRECTED: Request a TextCortex for handling raw text input.
      const config = {
        dimensions: 32,
        cortex_configs: {
            "text_input": { "type": "TextCortex" }
        }
      };
      const response = await api.createAgent(config);
      setStatusMessage(`New agent created: ${response.data.id}`);
      await fetchAgents();
      setSelectedAgent(response.data.id);
    } catch (error) {
      console.error("Failed to create agent:", error);
      setStatusMessage('Failed to create agent.');
    }
  };

  const handleLearn = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!selectedAgent || !textInput) return;
    try {
      setStatusMessage(`Agent is learning pattern: \"${textInput}\"...`);
      // CORRECTED: Use the updated learnWithAgent function.
      await api.learnWithAgent(selectedAgent, textInput);
      setTextInput('');
      setStatusMessage('Learning complete. Refreshing structure...');

      const response = await api.getAgentStructure(selectedAgent);
      setGraphData(response.data);
      setStatusMessage(`Visualizing agent: ${selectedAgent}`);
    } catch (error) {
      console.error("Failed to learn pattern:", error);
      setStatusMessage('Failed to learn pattern.');
    }
  };

  const handleDeleteAgent = async () => {
    if (!selectedAgent) return;
    if (window.confirm(`Are you sure you want to delete agent ${selectedAgent}?`)) {
      try {
        setStatusMessage(`Deleting agent ${selectedAgent}...`);
        await api.deleteAgent(selectedAgent);
        setStatusMessage(`Agent ${selectedAgent} deleted.`);
        setSelectedAgent(null);
        await fetchAgents();
      } catch (error) {
        console.error("Failed to delete agent:", error);
        setStatusMessage('Failed to delete agent.');
      }
    }
  };

  const handleStartTraining = async () => {
    if (!selectedAgent || !selectedEnv) return;
    try {
      setStatusMessage(`Starting training for ${selectedAgent} in ${selectedEnv}...`);
      await api.startTraining(selectedAgent, selectedEnv);
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
          <div className="control-group">
            <h4>Agent Management</h4>
            <div className="network-selector">
              <select onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSelectedAgent(e.target.value)} value={selectedAgent || ''}>
                <option value="" disabled>Select an Agent</option>
                {agents.map(agent => (
                  <option key={agent.id} value={agent.id}>{agent.id}</option>
                ))}
              </select>
              <button onClick={handleCreateAgent}>Create New Agent</button>
              <button onClick={handleDeleteAgent} disabled={!selectedAgent} className="delete-button">
                Delete Selected
              </button>
            </div>
          </div>

          {selectedAgent && (
            <div className="actions">
              <div className="control-group">
                <h4>Cognitive Learning</h4>
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
          )}
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
