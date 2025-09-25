import React, { useEffect, useState } from 'react';

interface ProgressData {
  current_step: number;
  total_steps: number;
  percentage: number;
  stage: string;
  eta_seconds?: number;
}

interface ProgressOverlayProps {
  generationId: string | null;
  onComplete: () => void;
  onError: (error: string) => void;
}

const ProgressOverlay: React.FC<ProgressOverlayProps> = ({ 
  generationId, 
  onComplete, 
  onError 
}) => {
  const [progress, setProgress] = useState<ProgressData>({
    current_step: 0,
    total_steps: 1,
    percentage: 0,
    stage: 'initializing'
  });
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!generationId) return;

    const eventSource = new EventSource(`/api/progress/${generationId}`);

    eventSource.onopen = () => {
      setIsConnected(true);
      console.log('Progress stream connected');
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'connected':
            console.log('Connected to progress stream');
            break;
            
          case 'progress':
            setProgress({
              current_step: data.current_step,
              total_steps: data.total_steps,
              percentage: data.percentage,
              stage: data.stage,
              eta_seconds: data.eta_seconds
            });
            break;
            
          case 'completed':
            console.log('Generation completed');
            eventSource.close();
            onComplete();
            break;
            
          case 'timeout':
            console.log('Progress stream timed out');
            eventSource.close();
            onError('Generation timed out. Please try again.');
            break;
            
          default:
            console.log('Unknown message type:', data.type);
        }
      } catch (error) {
        console.error('Error parsing progress data:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      setIsConnected(false);
      eventSource.close();
      onError('Connection lost. Please try again.');
    };

    // Cleanup function
    return () => {
      eventSource.close();
    };
  }, [generationId, onComplete, onError]);

  const formatETA = (seconds?: number) => {
    if (!seconds || seconds <= 0) return null;
    
    if (seconds < 60) {
      return `${seconds}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds}s`;
    }
  };

  const getStageDisplay = (stage: string) => {
    switch (stage) {
      case 'initializing':
        return 'Initializing...';
      case 'loading model':
        return 'Loading AI model...';
      case 'preprocessing sketch':
        return 'Processing your sketch...';
      case 'starting generation':
        return 'Starting generation...';
      case 'generating':
        return 'Generating image...';
      case 'saving results':
        return 'Saving results...';
      case 'error':
        return 'Error occurred';
      default:
        if (stage.includes('generating image')) {
          return stage.charAt(0).toUpperCase() + stage.slice(1) + '...';
        }
        return stage.charAt(0).toUpperCase() + stage.slice(1) + '...';
    }
  };

  if (!generationId) return null;

  return (
    <div className="loading-overlay">
      <div className="progress-container">
        {/* Progress Circle */}
        <div className="progress-circle">
          <svg width="120" height="120" viewBox="0 0 120 120">
            {/* Background circle */}
            <circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke="rgba(255, 255, 255, 0.2)"
              strokeWidth="8"
            />
            {/* Progress circle */}
            <circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke="var(--yellow)"
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 50}`}
              strokeDashoffset={`${2 * Math.PI * 50 * (1 - progress.percentage / 100)}`}
              transform="rotate(-90 60 60)"
              style={{ transition: 'stroke-dashoffset 0.3s ease' }}
            />
          </svg>
          {/* Percentage text */}
          <div className="progress-percentage">
            {progress.percentage}%
          </div>
        </div>

        {/* Progress Info */}
        <div className="progress-info">
          <h3 className="progress-title">{getStageDisplay(progress.stage)}</h3>
          
          <div className="progress-details">
            <div className="step-info">
              Step {progress.current_step} of {progress.total_steps}
            </div>
          </div>

          {/* Linear progress bar as backup */}
          <div className="progress-bar-container">
            <div className="progress-bar">
              <div 
                className="progress-bar-fill"
                style={{ 
                  width: `${progress.percentage}%`,
                  transition: 'width 0.3s ease'
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProgressOverlay;