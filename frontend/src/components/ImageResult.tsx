import React, { useEffect, useState } from 'react';
import { GenerationResult } from '../types';
import { checkGenerationStatus } from '../services/api';
import { useReset } from '../ResetContext';
import Icon from './Icon';

interface ProgressData {
  current_step: number;
  total_steps: number;
  percentage: number;
  stage: string;
  eta_seconds?: number;
  intermediate_image?: string;
}

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
  const [isMaximized, setIsMaximized] = useState(false);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const { resetTrigger } = useReset();

  useEffect(() => {
    if (resetTrigger > 0) {
      setResult(null);
      setGenerationResult(null);
      setSelectedImageIndex(0);
      setImageUrls([]);
      setProgress(null);
      setIsConnected(false);
    }
  }, [resetTrigger, setGenerationResult]);

  useEffect(() => {
    if (generationResult) {
      setResult(generationResult);
      setSelectedImageIndex(0);
      setImageUrls([]);
      setProgress(null);
      
      if (generationResult.status === 'processing') {
        setIsLoading(false); // Turn off any old loading overlay
        startProgressStream(generationResult.generation_id);
      } else if (generationResult.status === 'completed' && generationResult.image_url) {
        generateImageUrls(generationResult.generation_id, generationResult.image_url);
      }
    }
  }, [generationResult, setIsLoading]);

  const startProgressStream = (generationId: string) => {
    const apiUrl = process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '') || 'http://localhost:8000';
    const eventSource = new EventSource(`${apiUrl}/api/progress/${generationId}`);

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
              eta_seconds: data.eta_seconds,
              intermediate_image: data.intermediate_image
            });
            break;
            
          case 'completed':
            console.log('Generation completed');
            eventSource.close();
            setProgress(null);
            handleProgressComplete(generationId);
            break;
            
          case 'timeout':
            console.log('Progress stream timed out');
            eventSource.close();
            setProgress(null);
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
    };
  };

  const handleProgressComplete = async (generationId: string) => {
    try {
      const updatedResult = await checkGenerationStatus(generationId);
      setResult(updatedResult);
      
      if (updatedResult.status === 'completed' && updatedResult.image_url) {
        generateImageUrls(updatedResult.generation_id, updatedResult.image_url);
      }
    } catch (error) {
      console.error('Error checking final status:', error);
    }
  };

  const generateImageUrls = (generationId: string, firstImageUrl: string) => {
    // Check if this is a multi-image result (ends with _1)
    if (firstImageUrl.includes('_1')) {
      // Generate URLs for 3 images
      const urls = [];
      for (let i = 1; i <= 3; i++) {
        urls.push(`/api/images/${generationId}_${i}`);
      }
      setImageUrls(urls);
    } else {
      // Single image result
      setImageUrls([firstImageUrl]);
    }
  };

  const handleDownload = () => {
    if (imageUrls[selectedImageIndex]) {
      const imageUrl = `${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${imageUrls[selectedImageIndex]}`;
      
      const link = document.createElement('a');
      link.href = imageUrl;
      link.download = `generated-image-${selectedImageIndex + 1}-${new Date().getTime()}.png`;
      
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

  const handleThumbnailClick = (index: number) => {
    setSelectedImageIndex(index);
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

  // Show progress within placeholder during generation
  if (progress && result?.generation_id) {
    return (
      <div className="image-result progress">
        <div className="progress-container">
          {/* Always try to show intermediate image if available, even for early steps */}
          {progress.intermediate_image ? (
            <div className="intermediate-image-container">
              <img 
                src={progress.intermediate_image}
                alt="Generation in progress"
                className="intermediate-image"
              />
              <div className="progress-overlay">
                <div className="progress-info">
                  <div className="progress-text">{getStageDisplay(progress.stage)}</div>
                  <div className="progress-details">
                    Step {progress.current_step} of {progress.total_steps} ({progress.percentage}%)
                  </div>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-bar-fill"
                    style={{ width: `${progress.percentage}%` }}
                  />
                </div>
              </div>
            </div>
          ) : (
            <div className="progress-placeholder">
              <div className="progress-spinner"></div>
              <div className="progress-info">
                <div className="progress-text">{getStageDisplay(progress.stage)}</div>
                <div className="progress-details">
                  {progress.current_step > 0 ? (
                    `Step ${progress.current_step} of ${progress.total_steps} (${progress.percentage}%)`
                  ) : (
                    'Preparing...'
                  )}
                </div>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-bar-fill"
                  style={{ width: `${progress.percentage}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }
  
  // Empty state
  if (!result || (!progress && result.status === 'processing')) {
    return (
      <div className="image-result empty">
        <div className="placeholder">
          <p>your generated image will appear here</p>
        </div>
      </div>
    );
  }
  
  // Completed state with generated images
  return (
    <>
      <div className="image-result completed">
        {imageUrls.length > 0 && (
          <div className="image-gallery">
            {/* Main Image Display */}
            <div className="main-image-container">
              <div className="main-image-wrapper">
                <img 
                  src={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${imageUrls[selectedImageIndex]}`} 
                  alt="Generated from sketch" 
                  className="main-image"
                />
                
                {/* Result Controls - now inside main image wrapper */}
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
              </div>
            </div>

            {/* Thumbnail Gallery - only show if multiple images */}
            {imageUrls.length > 1 && (
              <div className="thumbnail-gallery">
                {imageUrls.map((imageUrl, index) => (
                  <div 
                    key={index}
                    className={`thumbnail-container ${selectedImageIndex === index ? 'selected' : ''}`}
                    onClick={() => handleThumbnailClick(index)}
                  >
                    <img 
                      src={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${imageUrl}`} 
                      alt={`Generated variation ${index + 1}`}
                      className="thumbnail-image"
                    />
                    <div className="thumbnail-overlay">
                      <span className="thumbnail-number">{index + 1}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {isMaximized && imageUrls.length > 0 && (
        <div className="image-modal-overlay" onClick={handleCloseMaximized}>
          <div className="image-modal-container">
            <button 
              className="image-modal-close"
              onClick={handleCloseMaximized}
              title="Close"
              aria-label="Close maximized view"
            >
              <Icon name="close" size={20} />
            </button>
            <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
              <img 
                src={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${imageUrls[selectedImageIndex]}`} 
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