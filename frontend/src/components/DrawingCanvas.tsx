import React, { useRef, useEffect } from 'react';
import { Excalidraw, exportToBlob } from '@excalidraw/excalidraw';
import { generateImage } from '../services/api';
import { Style, Model, GenerationResult } from '../types';
import { useReset } from '../ResetContext';
import Icon from './Icon';

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
  showCanvasOnly?: boolean;
  showControlsOnly?: boolean;
}

// Global shared reference to maintain state across components
let globalExcalidrawRef: any = null;
let globalSetExcalidrawRef: ((ref: any) => void) | null = null;

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
  showCanvasOnly = false,
  showControlsOnly = false,
}) => {
  const excalidrawRef = useRef<any>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const { triggerReset } = useReset();

  // Set up the global reference when canvas component mounts
  useEffect(() => {
    if (showCanvasOnly && excalidrawRef.current) {
      globalExcalidrawRef = excalidrawRef.current;
      // Notify any waiting controls
      if (globalSetExcalidrawRef) {
        globalSetExcalidrawRef(excalidrawRef.current);
      }
    }
  }, [showCanvasOnly, excalidrawRef.current]);

  // For controls component, wait for canvas reference
  useEffect(() => {
    if (showControlsOnly) {
      globalSetExcalidrawRef = (ref: any) => {
        globalExcalidrawRef = ref;
      };
    }
  }, [showControlsOnly]);

  const handleGenerateImage = async () => {
    const excalidrawInstance = globalExcalidrawRef || excalidrawRef.current;
    
    if (!excalidrawInstance) {
      showNotification('Canvas is not initialized. Please wait for the canvas to load.', 'error');
      return;
    }

    const elements = excalidrawInstance.getSceneElements();
    if (!elements || elements.length === 0) {
      showNotification('please draw something first', 'error');
      return;
    }

    if (!selectedStyle) {
      showNotification('please select a style first', 'error');
      return;
    }

    if (!selectedModel) {
      showNotification('please select a model first', 'error');
      return;
    }

    if (!description.trim()) {
      showNotification('please describe what you want to generate', 'error');
      return;
    }

    try {
      setIsLoading(true);

      const appState = excalidrawInstance.getAppState();

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
    const excalidrawInstance = globalExcalidrawRef || excalidrawRef.current;
    
    if (excalidrawInstance) {
      excalidrawInstance.resetScene();
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

  // Render only controls
  if (showControlsOnly) {
    return (
      <div className="canvas-controls">
        <button 
          className="generate-button"
          onClick={handleGenerateImage}
        >
          <Icon name="generate" size={16} />
          generate image
        </button>
        <button 
          className="reset-button"
          onClick={handleReset}
        >
          <Icon name="reset" size={16} />
          reset all
        </button>
      </div>
    );
  }

  // Render only canvas
  if (showCanvasOnly) {
    return (
      <div className="drawing-canvas">
        <div 
          ref={wrapperRef}
          className="excalidraw-wrapper" 
          style={{ 
            width: '100%', 
            height: '100%',
            minHeight: 400,
            border: 'none',
            borderRadius: '12px',
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
      </div>
    );
  }

  // Default render (both canvas and controls) - for backward compatibility
  return (
    <div className="drawing-canvas">
      <div 
        ref={wrapperRef}
        className="excalidraw-wrapper" 
        style={{ 
          width: '100%', 
          height: 400, 
          border: 'none',
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
          <Icon name="generate" size={16} />
          generate image
        </button>
        <button 
          className="reset-button"
          onClick={handleReset}
        >
          <Icon name="reset" size={16} />
          reset all
        </button>
      </div>
    </div>
  );
};

export default DrawingCanvas;