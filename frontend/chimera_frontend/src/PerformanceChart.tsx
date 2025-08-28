import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrainingMetric {
  episode: number;
  reward: number;
  avg_reward: number;
  epsilon: number;
  policy_loss: number;
}

interface PerformanceChartProps {
  data: TrainingMetric[];
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ data }) => {
  // Sample data for development and display if no real data is provided
  const sampleData: TrainingMetric[] = [
    { episode: 1, reward: -200, avg_reward: -200, epsilon: 0.9, policy_loss: 0.5 },
    { episode: 2, reward: -190, avg_reward: -195, epsilon: 0.88, policy_loss: 0.48 },
    { episode: 3, reward: -150, avg_reward: -180, epsilon: 0.86, policy_loss: 0.45 },
    { episode: 4, reward: -120, avg_reward: -165, epsilon: 0.84, policy_loss: 0.42 },
    { episode: 5, reward: -100, avg_reward: -152, epsilon: 0.82, policy_loss: 0.38 },
  ];

  const chartData = data.length > 0 ? data : sampleData;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={chartData}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#555" />
        <XAxis dataKey="episode" stroke="#ccc" />
        <YAxis yAxisId="left" stroke="#8884d8" />
        <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
        <Tooltip
          contentStyle={{ backgroundColor: '#222', border: '1px solid #555' }}
          labelStyle={{ color: '#fff' }}
        />
        <Legend />
        <Line yAxisId="left" type="monotone" dataKey="avg_reward" name="Average Reward" stroke="#8884d8" activeDot={{ r: 8 }} />
        <Line yAxisId="right" type="monotone" dataKey="policy_loss" name="Policy Loss" stroke="#82ca9d" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default PerformanceChart;
