import React from 'react';

export default function PredictionProgress({ predictionEvent, isBusy }) {
  if (!isBusy || !predictionEvent) return null;

  const getStageIcon = (type, stage) => {
    if (type === 'started') return '🚀';
    if (type === 'error') return '❌';
    if (type === 'complete') return '✅';
    if (type === 'progress') {
        if (stage === 'preprocessing') return '🔄';
        if (stage === 'inference') return '🧠';
        if (stage === 'postprocessing') return '📊';
    }
    return '⏳';
  };

  const getMessage = () => {
    if (predictionEvent.type === 'started') {
        return `Starting prediction with ${predictionEvent.model}...`;
    }
    if (predictionEvent.type === 'error') {
        return `Error: ${predictionEvent.error}`;
    }
    if (predictionEvent.type === 'complete') {
        return `Prediction complete!`;
    }
    if (predictionEvent.type === 'progress') {
        return predictionEvent.detail || `Running ${predictionEvent.stage}...`;
    }
    return 'Analyzing...';
  };

  return (
    <div className="prediction-progress" style={{
        margin: '15px 0', 
        padding: '12px 16px', 
        background: 'var(--surface-sunken)', 
        borderRadius: '8px',
        border: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
    }}>
      <div className="spinner-icon" style={{width: '20px', height: '20px'}} />
      <div>
        <div style={{fontWeight: '500', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px'}}>
            <span>{getStageIcon(predictionEvent.type, predictionEvent.stage)}</span>
            <span>{getMessage()}</span>
        </div>
      </div>
    </div>
  );
}
