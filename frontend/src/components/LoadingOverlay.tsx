import React from 'react';
import '../styles/LoadingOverlay.css';

interface LoadingOverlayProps {
  modelName?: string;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ modelName }) => {
  return (
    <div className="loading-overlay">
      <div className="spinner"></div>
      <p className="loading-text">Generating your image...</p>
      {modelName ? (
        <p className="loading-model">Using {modelName}</p>
      ) : null}
      <p className="loading-subtext">This may take 10-30 seconds depending on the selected model</p>
    </div>
  );
};

export default LoadingOverlay;