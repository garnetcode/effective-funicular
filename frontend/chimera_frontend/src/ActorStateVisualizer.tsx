import React from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ActorState {
  h_t: number[];
  z_t: number[];
  epsilon: number;
  action_probs: number[];
}

interface ActorStateVisualizerProps {
  actorState: ActorState | null;
}

const ActorStateVisualizer: React.FC<ActorStateVisualizerProps> = ({ actorState }) => {
  if (!actorState) {
    return <div className="visualizer-placeholder" style={{color: 'white', margin: '20px'}}>Waiting for actor state...</div>;
  }

  const { h_t, epsilon, action_probs } = actorState;

  const actionProbData = action_probs.map((prob, index) => ({
    action: `Action ${index}`,
    probability: prob,
  }));

  const hStateData = h_t.map((value, index) => ({
    index: index,
    value: value,
  }));

  return (
    <div style={{ color: 'white', padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ textAlign: 'center', margin: '0 0 10px 0' }}>Actor State</h3>

      <div style={{ marginBottom: '10px' }}>
        <strong>Epsilon:</strong> {epsilon.toFixed(4)}
      </div>

      <div style={{ flex: 1, marginBottom: '10px' }}>
        <h4 style={{ margin: '0 0 5px 0' }}>Action Probabilities</h4>
        <ResponsiveContainer width="100%" height="90%">
          <BarChart data={actionProbData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="action" stroke="#888" tick={false} />
            <YAxis stroke="#888" domain={[0, 1]}/>
            <Tooltip contentStyle={{ backgroundColor: '#222', border: '1px solid #444' }} />
            <Bar dataKey="probability" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ flex: 1 }}>
        <h4 style={{ margin: '0 0 5px 0' }}>Hidden State (h_t)</h4>
        <ResponsiveContainer width="100%" height="90%">
          <LineChart data={hStateData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="index" stroke="#888" />
            <YAxis stroke="#888" />
            <Tooltip contentStyle={{ backgroundColor: '#222', border: '1px solid #444' }} />
            <Line type="monotone" dataKey="value" stroke="#82ca9d" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ActorStateVisualizer;
