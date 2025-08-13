import React, { useState, useEffect, useCallback } from 'react';
import * as api from './api';
import NetworkVisualizer from './NetworkVisualizer';
import './App.css';

function App() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [textInput, setTextInput] = useState('');
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

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

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
        </div>
        {selectedAgent && (
          <div className="actions">
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
        )}
        <div className="status-bar">
          {statusMessage}
        </div>
      </div>
      <main className="visualizer-container">
        <NetworkVisualizer graphData={graphData} />
      </main>
    </div>
  );
}

export default App;
