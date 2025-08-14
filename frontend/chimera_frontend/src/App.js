import React, { useState, useEffect, useCallback } from 'react';
import * as api from './api';
import NetworkVisualizer from './NetworkVisualizer';
import './App.css';

function App() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [environments, setEnvironments] = useState([]);
  const [selectedEnv, setSelectedEnv] = useState('');
  const [graphData, setGraphData] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [trainingMetrics, setTrainingMetrics] = useState([]);
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

  const handleLearn = async (e) => {
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

  useEffect(() => {
    if (!selectedAgent) return;

    const ws_scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const ws_path = `${ws_scheme}://${window.location.host.replace('3000', '8000')}/ws/training/${selectedAgent}/`;

    const socket = new WebSocket(ws_path);

    socket.onopen = () => {
      console.log("WebSocket connected for agent:", selectedAgent);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received metric:", data.message);
      setTrainingMetrics(prevMetrics => [...prevMetrics, data.message]);
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected for agent:", selectedAgent);
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    // Cleanup on component unmount or when selectedAgent changes
    return () => {
      socket.close();
    };
  }, [selectedAgent]);


  return (
    <div className="App">
      <header className="App-header">
        <h1>Project Chimera</h1>
      </header>
      <div className="controls">
        <div className="network-selector">
          <select onChange={(e) => setSelectedAgent(e.target.value)} value={selectedAgent || ''}>
            <option value="" disabled>Select an Agent</option>
            {agents.map(agent => (
              <option key={agent.id} value={agent.id}>{agent.id}</option>
            ))}
          </select>
          <button onClick={handleCreateAgent}>Create New Agent</button>
          <button onClick={handleDeleteAgent} disabled={!selectedAgent} className="delete-button">
            Delete Selected Agent
          </button>
        </div>
        {selectedAgent && (
          <div className="actions">
            <div className="action-item">
              <h4>Cognitive Learning</h4>
              <form onSubmit={handleLearn}>
                <input
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter a pattern to learn"
                />
                <button type="submit">Learn & Organize</button>
              </form>
            </div>
            <div className="action-item">
              <h4>Reinforcement Learning</h4>
              <select onChange={(e) => setSelectedEnv(e.target.value)} value={selectedEnv}>
                {environments.map(env => (
                  <option key={env.id} value={env.id}>{env.name}</option>
                ))}
              </select>
              <button onClick={handleStartTraining} disabled={!selectedEnv}>
                Start Training
              </button>
              <div className="metrics-display">
                {/* Metrics will be displayed here */}
              </div>
            </div>
          </div>
        )}
        <div className="status-bar">
          {statusMessage}
        </div>
      </div>
      <main className="visualizer-container">
        <NetworkVisualizer graphData={graphData} />
        <div className="metrics-container">
          <h3>Training Metrics</h3>
          <ul>
            {trainingMetrics.map((metric, index) => (
              <li key={index}>
                Episode {metric.episode}: Reward = {metric.total_reward.toFixed(2)}, Avg Reward = {metric.avg_reward.toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      </main>
    </div>
  );
}

export default App;
