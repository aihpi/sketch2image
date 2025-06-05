import React, { useEffect, useState } from 'react';
import { GenerationResult } from '../types';
import { checkGenerationStatus } from '../services/api';
import '../styles/ImageResult.css';
import { useReset } from '../ResetContext';

interface ImageResultProps {
  generationResult: GenerationResult | null;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  setGenerationResult: React.Dispatch<React.SetStateAction<GenerationResult | null>>;
}

const ImageResult: React.FC<ImageResultProps> = ({ 
  generationResult, 
  setIsLoading,
  setGenerationResult 
}) => {
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
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
  
  if (!result) {
    return (
      <div className="image-result empty">
        <div className="placeholder">
          <p>Your generated image will appear here</p>
          <p>Draw a sketch and click "Generate Image" to start</p>
        </div>
      </div>
    );
  }
  
  if (result.status === 'processing') {
    return (
      <div className="image-result processing">
        <div className="status-message">
          <p>Processing your image...</p>
          <p className="info-text">This may take 10-30 seconds</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="image-result completed">
      {result.image_url && (
        <div className="image-container">
          <div className="image-wrapper">
            <img 
              src={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${result.image_url}`} 
              alt="Generated from sketch" 
            />
          </div>
          <div className="image-controls">
            <button 
              className="download-button"
              onClick={handleDownload}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="download-icon">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Download Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageResult;