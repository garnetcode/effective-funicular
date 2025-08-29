import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';

interface ActionVisualizerProps {
  probabilities: number[];
}

const ActionVisualizer: React.FC<ActionVisualizerProps> = ({ probabilities }) => {
  if (!probabilities || probabilities.length === 0) {
    return <div className="visualizer-placeholder" style={{color: 'white', margin: '20px'}}>Waiting for action data...</div>;
  }

  const chartData = probabilities.map((p, index) => ({
    name: `Action ${index}`,
    probability: p,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={chartData}
        margin={{
          top: 20,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="name" stroke="#888" angle={-45} textAnchor="end" height={50} />
        <YAxis stroke="#888" domain={[0, 1]} />
        <Tooltip
          contentStyle={{ backgroundColor: '#222', border: '1px solid #444' }}
          labelStyle={{ color: '#eee' }}
          formatter={(value: number) => value.toFixed(4)}
        />
        <Legend wrapperStyle={{ color: '#eee' }} />
        <Bar dataKey="probability" fill="#8884d8" isAnimationActive={true} animationDuration={500} />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default ActionVisualizer;
