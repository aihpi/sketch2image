import React from 'react';
import '../styles/LoadingOverlay.css';

const LoadingOverlay: React.FC = () => {
  return (
    <div className="loading-overlay">
      <div className="spinner"></div>
      <p className="loading-text">Generating your image...</p>
      <p className="loading-subtext">This may take 10-30 seconds</p>
    </div>
  );
};

export default LoadingOverlay;