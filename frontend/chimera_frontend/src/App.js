import React, { useState, useEffect, useCallback } from 'react';
import * as api from './api';
import NetworkVisualizer from './NetworkVisualizer';
import './App.css';

function App() {
  const [networks, setNetworks] = useState([]);
  const [selectedNetwork, setSelectedNetwork] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [statusMessage, setStatusMessage] = useState('');

  const fetchNetworks = useCallback(async () => {
    try {
      const response = await api.getNetworks();
      setNetworks(response.data);
    } catch (error) {
      console.error("Failed to fetch networks:", error);
      setStatusMessage('Failed to fetch networks.');
    }
  }, []);

  useEffect(() => {
    fetchNetworks();
  }, [fetchNetworks]);

  useEffect(() => {
    if (selectedNetwork) {
      api.getNetworkStructure(selectedNetwork)
        .then(response => {
          setGraphData(response.data);
          setStatusMessage(`Visualizing network: ${selectedNetwork}`);
        })
        .catch(error => {
          console.error("Failed to fetch graph structure:", error);
          setStatusMessage('Failed to fetch graph structure.');
        });
    } else {
      setGraphData(null);
    }
  }, [selectedNetwork]);

  const handleCreateNetwork = async () => {
    try {
      setStatusMessage('Creating new network...');
      const response = await api.createNetwork({ dimensions: 64 });
      setStatusMessage(`New network created: ${response.data.id}`);
      await fetchNetworks();
      setSelectedNetwork(response.data.id);
    } catch (error) {
      console.error("Failed to create network:", error);
      setStatusMessage('Failed to create network.');
    }
  };

  const handleLearn = async (e) => {
    e.preventDefault();
    if (!selectedNetwork || !textInput) return;
    try {
      setStatusMessage('Learning pattern...');
      await api.learnText(selectedNetwork, textInput);
      setTextInput('');
      setStatusMessage('Pattern learned. Organizing...');
      // Automatically trigger organization after learning
      const orgResponse = await api.organizeNetwork(selectedNetwork);
      setStatusMessage(orgResponse.data.status);
      // Refresh graph data
      const response = await api.getNetworkStructure(selectedNetwork);
      setGraphData(response.data);
    } catch (error) {
      console.error("Failed to learn pattern:", error);
      setStatusMessage('Failed to learn pattern.');
    }
  };

  const handleOrganize = async () => {
    if (!selectedNetwork) return;
    try {
        setStatusMessage('Organizing...');
        const orgResponse = await api.organizeNetwork(selectedNetwork);
        setStatusMessage(orgResponse.data.status);
        // Refresh graph data
        const response = await api.getNetworkStructure(selectedNetwork);
        setGraphData(response.data);
    } catch (error) {
        console.error("Failed to organize network:", error);
        setStatusMessage('Failed to organize network.');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Project Chimera</h1>
        <p>A Bio-Inspired Cognitive Architecture</p>
      </header>
      <div className="controls">
        <div className="network-selector">
          <select onChange={(e) => setSelectedNetwork(e.target.value)} value={selectedNetwork || ''}>
            <option value="" disabled>Select a Network</option>
            {networks.map(net => (
              <option key={net.id} value={net.id}>{net.id}</option>
            ))}
          </select>
          <button onClick={handleCreateNetwork}>Create New Network</button>
        </div>
        {selectedNetwork && (
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
            <button onClick={handleOrganize}>Organize Step</button>
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
