import React, { useEffect, useState } from 'react';
import { toast } from 'react-toastify';
import { GenerationResult } from '../types';
import { checkGenerationStatus } from '../services/api';
import '../styles/ImageResult.css';

interface ImageResultProps {
  generationResult: GenerationResult | null;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const ImageResult: React.FC<ImageResultProps> = ({ generationResult, setIsLoading }) => {
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Update local state when parent passes a new generation result
  useEffect(() => {
    if (generationResult) {
      setResult(generationResult);
      
      // If status is processing, start polling
      if (generationResult.status === 'processing') {
        startStatusPolling(generationResult.generation_id);
      }
    }
  }, [generationResult]);
  
  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);
  
  // Start polling for status updates
  const startStatusPolling = (generationId: string) => {
    // Clear any existing polling
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    
    // Set loading state
    setIsLoading(true);
    
    // Start polling every 2 seconds
    const interval = setInterval(async () => {
      try {
        const updatedResult = await checkGenerationStatus(generationId);
        setResult(updatedResult);
        
        // If generation is complete, stop polling
        if (updatedResult.status === 'completed') {
          clearInterval(interval);
          setPollingInterval(null);
          setIsLoading(false);
          toast.success('Image generation completed!');
        }
      } catch (error) {
        console.error('Error checking generation status:', error);
        clearInterval(interval);
        setPollingInterval(null);
        setIsLoading(false);
        toast.error('Failed to check generation status');
      }
    }, 2000);
    
    setPollingInterval(interval);
  };
  
  // If no result yet, show placeholder
  if (!result) {
    return (
      <div className="image-result empty">
        <div className="placeholder">
          <p>Your generated image will appear here</p>
          <p>Draw a sketch and click "Generate Image" to get started</p>
        </div>
      </div>
    );
  }
  
  // If processing, show status
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
  
  // If completed, show the image
  return (
    <div className="image-result completed">
      {result.image_url && (
        <div className="image-container">
          <img 
            src={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${result.image_url}`} 
            alt="Generated from sketch" 
          />
          <div className="image-controls">
            <a 
              href={`${process.env.REACT_APP_API_URL?.replace(/\/api\/?$/, '')}${result.image_url}`}
              download="generated-image.png"
              className="download-button"
            >
              Download Image
            </a>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageResult;