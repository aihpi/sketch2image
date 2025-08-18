import React, { useEffect, useState } from 'react';
import { GenerationResult } from '../types';
import { checkGenerationStatus } from '../services/api';
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
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const { resetTrigger } = useReset();

  useEffect(() => {
    if (resetTrigger > 0) {
      setResult(null);
      setGenerationResult(null);
      setSelectedImageIndex(0);
      setImageUrls([]);
    }
  }, [resetTrigger, setGenerationResult]);

  useEffect(() => {
    if (generationResult) {
      setResult(generationResult);
      setSelectedImageIndex(0); // Reset to first image when new results come in
      setImageUrls([]); // Clear previous URLs
      
      if (generationResult.status === 'processing') {
        startStatusPolling(generationResult.generation_id);
      } else if (generationResult.status === 'completed' && generationResult.image_url) {
        // Generate URLs for multiple images
        generateImageUrls(generationResult.generation_id, generationResult.image_url);
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
          
          if (updatedResult.image_url) {
            generateImageUrls(generationId, updatedResult.image_url);
          }
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
  
  if (!result || result.status === 'processing') {
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