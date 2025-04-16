import React, { useRef, useEffect } from 'react';
import { Excalidraw, exportToBlob } from '@excalidraw/excalidraw';
import { toast } from 'react-toastify';
import { generateImage } from '../services/api';
import { Style, GenerationResult } from '../types';

interface DrawingCanvasProps {
  selectedStyle: Style | null;
  description: string;
  setDescription: React.Dispatch<React.SetStateAction<string>>;
  setGenerationResult: React.Dispatch<React.SetStateAction<GenerationResult | null>>;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  selectedStyle,
  description,
  setDescription,
  setGenerationResult,
  setIsLoading,
}) => {
  const excalidrawRef = useRef<any>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Force Excalidraw to resize properly
  useEffect(() => {
    const handleResize = () => {
      window.dispatchEvent(new Event('resize'));
    };

    // Force an initial resize
    setTimeout(handleResize, 200);

    // Add listener for window resize
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleGenerateImage = async () => {
    if (!selectedStyle) {
      toast.error('Please select a style first');
      return;
    }

    if (!excalidrawRef.current) {
      toast.error('Canvas is not initialized');
      return;
    }

    const elements = excalidrawRef.current.getSceneElements();
    if (!elements || elements.length === 0) {
      toast.error('Please draw something first');
      return;
    }

    try {
      setIsLoading(true);

      // Get the scene elements and app state
      const appState = excalidrawRef.current.getAppState();

      // Create a modified app state for export
      const exportAppState = {
        ...appState,
        exportWithDarkMode: false,
        exportBackground: true,
        viewBackgroundColor: '#ffffff',
      };

      // Export the canvas as PNG
      const blob = await exportToBlob({
        elements,
        appState: exportAppState,
        mimeType: 'image/png',
        files: null,
      });

      // Convert blob to file
      const file = new File([blob], 'sketch.png', { type: 'image/png' });

      // Send to API for processing
      const result = await generateImage(file, selectedStyle.id, description);

      // Set the generation result
      setGenerationResult(result);
      
      // Success notification
      toast.success('Image generation started!');
    } catch (error) {
      console.error('Failed to generate image:', error);
      toast.error('Failed to generate image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearCanvas = () => {
    if (excalidrawRef.current) {
      excalidrawRef.current.resetScene();
    }
  };

  return (
    <div className="drawing-canvas">
      <div 
        ref={wrapperRef}
        className="excalidraw-wrapper" 
        style={{ 
          width: '100%', 
          height: 500, 
          border: '1px solid #ddd',
          borderRadius: '4px',
          overflow: 'hidden',
          position: 'relative'
        }}
      >
        <Excalidraw
          ref={excalidrawRef}
          initialData={{
            appState: { 
              viewBackgroundColor: '#ffffff',
            },
          }}
          // The following props ensure the Excalidraw component takes up all the space in its container
          UIOptions={{
            canvasActions: {
              changeViewBackgroundColor: false,
              export: false,
              loadScene: false,
              saveToActiveFile: false,
              saveAsImage: false,
              toggleTheme: false,
            }
          }}
        />
      </div>
      
      <div className="canvas-controls">
        <button 
          className="generate-button"
          onClick={handleGenerateImage}
          disabled={!selectedStyle}
        >
          Generate Image
        </button>
        <button 
          className="clear-button"
          onClick={handleClearCanvas}
        >
          Clear Canvas
        </button>
      </div>
    </div>
  );
};

export default DrawingCanvas;