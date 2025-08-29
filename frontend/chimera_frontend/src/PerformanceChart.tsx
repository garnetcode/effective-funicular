import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrainingMetric {
  episode: number;
  reward: number;
  avg_reward: number;
  epsilon: number;
  policy_loss: number;
  total_steps: number;
}

interface PerformanceChartProps {
  data: TrainingMetric[];
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="visualizer-placeholder" style={{color: 'white', margin: '20px'}}>Waiting for training metrics...</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="episode" stroke="#888" name="Episode" />
        <YAxis yAxisId="left" stroke="#8884d8" label={{ value: 'Avg Reward', angle: -90, position: 'insideLeft', fill: '#8884d8' }} />
        <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" label={{ value: 'Policy Loss', angle: -90, position: 'insideRight', fill: '#82ca9d' }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#222', border: '1px solid #444' }}
          labelStyle={{ color: '#eee' }}
        />
        <Legend wrapperStyle={{ color: '#eee' }} />
        <Line yAxisId="left" type="monotone" dataKey="avg_reward" name="Avg Reward" stroke="#8884d8" dot={false} isAnimationActive={true} animationDuration={500} animationEasing="ease-out" />
        <Line yAxisId="right" type="monotone" dataKey="policy_loss" name="Policy Loss" stroke="#82ca9d" dot={false} isAnimationActive={true} animationDuration={500} animationEasing="ease-out" />
        <Line yAxisId="right" type="monotone" dataKey="epsilon" name="Epsilon" stroke="#ffc658" dot={false} isAnimationActive={true} animationDuration={500} animationEasing="ease-out" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default PerformanceChart;
