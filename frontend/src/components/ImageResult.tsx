import React, { useEffect, useState } from 'react';
import { GenerationResult } from '../types';
import { checkGenerationStatus } from '../services/api';
// import '../styles/ImageResult.css';
import { useReset } from '../ResetContext';
import Icon from './Icon';

interface ImageResultProps {
  generationResult: GenerationResult | null;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setGenerationResult: React.Dispatch<React.SetStateAction<GenerationResult | null>>;
  onRegenerate?: () => void;
}

const ImageResult: React.FC<ImageResultProps> = ({ 
  generationResult, 
  setIsLoading,
  setGenerationResult,
  onRegenerate 
}) => {
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  const [isMaximized, setIsMaximized] = useState(false);
  const { resetTrigger } = useReset();

  useEffect(() => {
    if (resetTrigger > 0) {
      setResult(null);
      setGenerationResult(null);
    }
  }, [resetTrigger, setGenerationResult]);

  useEffect(() => {
    if (generationResult) {
      setResult(generationResult);
      
      if (generationResult.status === 'processing') {
        startStatusPolling(generationResult.generation_id);
      }
    }
  }, [generationResult]);
  
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);
  
  const startStatusPolling = (generationId: string) => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    
    setIsLoading(true);
    
    const interval = setInterval(async () => {
      try {
        const updatedResult = await checkGenerationStatus(generationId);
        setResult(updatedResult);
        
        if (updatedResult.status === 'completed') {
          clearInterval(interval);
          setPollingInterval(null);
          setIsLoading(false);
        }
      } catch (error) {
        console.error('Error checking generation status:', error);
        clearInterval(interval);
        setPollingInterval(null);
        setIsLoading(false);
      }
    }, 2000);
    
    setPollingInterval(interval);
  };

  const handleDownload = () => {
    if (result?.image_url) {
      const imageUrl = `${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${result.image_url}`;
      
      const link = document.createElement('a');
      link.href = imageUrl;
      link.download = `generated-image-${new Date().getTime()}.png`;
      
      fetch(imageUrl)
        .then(response => response.blob())
        .then(blob => {
          const url = URL.createObjectURL(blob);
          link.href = url;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        })
        .catch(error => {
          console.error('Error downloading image:', error);
        });
    }
  };

  const handleMaximize = () => {
    setIsMaximized(true);
  };

  const handleCloseMaximized = () => {
    setIsMaximized(false);
  };

  // Handle escape key to close maximized view
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMaximized) {
        setIsMaximized(false);
      }
    };

    if (isMaximized) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isMaximized]);
  
  if (!result) {
    return (
      <>
        <div className="image-result empty">
          <div className="placeholder">
            <p>your generated image will appear here</p>
          </div>
        </div>
      </>
    );
  }
  
  if (result.status === 'processing') {
    return (
      <>
        <div className="image-result processing">
          <div className="status-message">
            <p>generating your image...</p>
            <p className="info-text">this may take 10-30 seconds</p>
          </div>
        </div>
      </>
    );
  }
  
  return (
    <>
      <div className="image-result completed">
        {result.image_url && (
          <div className="image-container">
            <div className="image-wrapper">
              <img 
                src={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${result.image_url}`} 
                alt="Generated from sketch" 
              />
            </div>
          </div>
        )}
      </div>

      {/* Result Controls - rendered in parent component */}
      {result?.image_url && (
        <div className="result-controls">
          <button 
            className="control-button"
            onClick={handleDownload}
            title="Download Image"
            aria-label="Download Image"
          >
            <Icon name="download" size={16} />
            download
          </button>
          
          <button 
            className="control-button"
            onClick={handleMaximize}
            title="View Full Size"
            aria-label="View Full Size"
          >
            <Icon name="expand" size={16} />
            view full size
          </button>
        </div>
      )}

      {/* Maximized Image Modal */}
      {isMaximized && result?.image_url && (
        <div className="image-modal-overlay" onClick={handleCloseMaximized}>
          <div className="image-modal-container">
            <button 
              className="image-modal-close"
              onClick={handleCloseMaximized}
              title="Close"
              aria-label="Close maximized view"
            >
              ×
            </button>
            <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
              <img 
                src={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${result.image_url}`} 
                alt="Generated from sketch - Maximized view" 
                className="maximized-image"
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ImageResult;