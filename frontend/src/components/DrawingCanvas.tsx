import React, { useRef, useEffect } from 'react';
import { Excalidraw, exportToBlob } from '@excalidraw/excalidraw';
import { generateImage } from '../services/api';
import { Style, Model, GenerationResult } from '../types';
import { useReset } from '../ResetContext';

interface DrawingCanvasProps {
  selectedStyle: Style | null;
  selectedModel: Model | null;
  styles: Style[];
  models: Model[];
  description: string;
  setDescription: React.Dispatch<React.SetStateAction<string>>;
  setSelectedStyle: React.Dispatch<React.SetStateAction<Style | null>>;
  setSelectedModel: React.Dispatch<React.SetStateAction<Model | null>>;
  setGenerationResult: React.Dispatch<React.SetStateAction<GenerationResult | null>>;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  showNotification: (message: string, type: 'error' | 'info' | 'success') => void;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  selectedStyle,
  selectedModel,
  styles,
  models,
  description,
  setDescription,
  setSelectedStyle,
  setSelectedModel,
  setGenerationResult,
  setIsLoading,
  showNotification,
}) => {
  const excalidrawRef = useRef<any>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const { triggerReset } = useReset();

  useEffect(() => {
    const handleResize = () => {
      window.dispatchEvent(new Event('resize'));
    };

    setTimeout(handleResize, 200);

    window.addEventListener('resize', handleResize);

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleGenerateImage = async () => {
    if (!excalidrawRef.current) {
      showNotification('Canvas is not initialized', 'error');
      return;
    }

    const elements = excalidrawRef.current.getSceneElements();
    if (!elements || elements.length === 0) {
      showNotification('Please draw something first', 'error');
      return;
    }

    if (!selectedStyle) {
      showNotification('Please select a style first', 'error');
      return;
    }

    if (!selectedModel) {
      showNotification('Please select a model first', 'error');
      return;
    }

    if (!description.trim()) {
      showNotification('Please describe what you want to generate', 'error');
      return;
    }

    try {
      setIsLoading(true);

      const appState = excalidrawRef.current.getAppState();

      const exportAppState = {
        ...appState,
        exportWithDarkMode: false,
        exportBackground: true,
        viewBackgroundColor: '#ffffff',
      };

      const blob = await exportToBlob({
        elements,
        appState: exportAppState,
        mimeType: 'image/png',
        files: null,
      });

      const file = new File([blob], 'sketch.png', { type: 'image/png' });

      const result = await generateImage(file, selectedStyle.id, selectedModel.id, description);

      setGenerationResult(result);
      
    } catch (error) {
      console.error('Failed to generate image:', error);
      showNotification('Failed to generate image. Please try again.', 'error');
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    if (excalidrawRef.current) {
      excalidrawRef.current.resetScene();
    }
    
    setDescription('');
    
    if (styles.length > 0) {
      setSelectedStyle(styles[0]);
    }
    
    if (models.length > 0) {
      setSelectedModel(models[0]);
    }

    setGenerationResult(null);
    
    triggerReset();
  };

  return (
    <div className="drawing-canvas">
      <div 
        ref={wrapperRef}
        className="excalidraw-wrapper" 
        style={{ 
          width: '100%', 
          height: 500, 
          border: '1px solid #e0e0e0',
          borderRadius: '8px',
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
        >
          Generate Image
        </button>
        <button 
          className="clear-button"
          onClick={handleReset}
        >
          Reset All
        </button>
      </div>
    </div>
  );
};

export default DrawingCanvas;