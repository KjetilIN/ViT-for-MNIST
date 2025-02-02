"use client"
import { BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

interface PredictionPanelProps {
  digit: number | null;
  probabilities: {[key: number]: number};
}

export default function PredictionPanel({ digit, probabilities }: PredictionPanelProps) {
  // Convert probabilities to recharts format
  const chartData = Object.entries(probabilities)
    .map(([digit, probability]) => ({
      digit: parseInt(digit),
      probability: probability * 100
    }))
    .sort((a, b) => b.probability - a.probability);

  return (
    <div className="bg-gray-900 text-white p-4 rounded-lg">
      {digit !== null ? (
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold">Predicted Digit: {digit}</h2>
        </div>
      ) : (
        <div className="text-center mb-4">
          <h2 className="text-2xl text-gray-500">No prediction yet</h2>
        </div>
      )}
      
      <BarChart 
        width={400} 
        height={300} 
        data={chartData}
        layout="horizontal"
      >
        <YAxis type="number" domain={[0, 100]} />
        <XAxis dataKey="digit" type="category" />
        <Tooltip 
          formatter={(value: number) => [`${value.toFixed(2)}%`, 'Probability']}
        />
        <Bar 
          dataKey="probability" 
          fill="#3B82F6" 
          barSize={20}
        />
      </BarChart>
    </div>
  );
}