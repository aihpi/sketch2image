// src/App.tsx
import React, { useState, useEffect } from 'react';
import './styles/App.css'; // This will be our single source of truth for styles

import DrawingCanvas from './components/DrawingCanvas';
import StyleSelector from './components/StyleSelector';
import ModelSelector from './components/ModelSelector';
import ImageResult from './components/ImageResult';
import LoadingOverlay from './components/LoadingOverlay';
import Notification from './components/Notification';
import { ResetProvider } from './ResetContext';

import { fetchStyles, fetchModels } from './services/api';

import { Style, Model, GenerationResult } from './types';

const App: React.FC = () => {
  const [styles, setStyles] = useState<Style[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedStyle, setSelectedStyle] = useState<Style | null>(null);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [description, setDescription] = useState<string>('');
  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [notification, setNotification] = useState<{
    show: boolean;
    message: string;
    type: 'error' | 'info' | 'success';
  }>({
    show: false,
    message: '',
    type: 'info'
  });

  const showNotification = (message: string, type: 'error' | 'info' | 'success') => {
    setNotification({
      show: true,
      message,
      type
    });
  };

  const closeNotification = () => {
    setNotification(prev => ({ ...prev, show: false }));
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const stylesData = await fetchStyles();
        setStyles(stylesData);
        if (stylesData.length > 0) {
          // Find and set a default style, e.g., 'photorealistic'
          const defaultStyle = stylesData.find(s => s.id === 'photorealistic') || stylesData[0];
          setSelectedStyle(defaultStyle);
        }

        const modelsData = await fetchModels();
        setModels(modelsData);
        if (modelsData.length > 0) {
          // Find and set a default model if possible
          const defaultModel = modelsData.find(m => m.id.includes('sdxl-1.0')) || modelsData[0];
          setSelectedModel(defaultModel);
        }
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
        showNotification('Failed to connect to the server. Please check your connection and reload.', 'error');
      }
    };

    fetchData();
  }, []);

  return (
    <ResetProvider>
      <div className="app">
        {notification.show && (
          <Notification
            message={notification.message}
            type={notification.type}
            onClose={closeNotification}
          />
        )}
        
        {/* LEFT COLUMN: Controls */}
        <div className="left-column">
          <header className="app-header">
            <h1>Sketch to Image</h1>
          </header>

          <div className="controls-wrapper">
            <div className="control-group">
              <ModelSelector 
                models={models} 
                selectedModel={selectedModel} 
                setSelectedModel={setSelectedModel} 
              />
            </div>

            <div className="control-group">
              <StyleSelector 
                styles={styles} 
                selectedStyle={selectedStyle} 
                setSelectedStyle={setSelectedStyle} 
              />
            </div>

            <div className="control-group">
              <label className="control-label" htmlFor="description">
                Description
              </label>
              <textarea
                id="description"
                className="modern-textarea"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe your vision... e.g., a cat wearing a tiny hat"
                required
                rows={3}
              />
            </div>

            <div className="generation-controls">
              <DrawingCanvas 
                selectedStyle={selectedStyle}
                selectedModel={selectedModel}
                styles={styles}
                models={models}
                description={description}
                setDescription={setDescription}
                setSelectedStyle={setSelectedStyle}
                setSelectedModel={setSelectedModel}
                setGenerationResult={setGenerationResult}
                setIsLoading={setIsLoading}
                showNotification={showNotification}
                showControlsOnly={true}
              />
            </div>
          </div>

          <footer className="app-footer">
            <div className="footer-content">
              <div className="footer-logo-container">
                <img 
                  src="/logos.jpg" 
                  alt="HPI AI Service Center Logo" 
                  className="footer-logos"
                />
                <p className="footer-text">
                  powered by <a href="https://hpi.de/en/research/hpi-data-center/ai-service-center/" className="footer-link" target="_blank" rel="noopener noreferrer">HPI's AI Service Center</a>
                </p>
              </div>
            </div>
          </footer>
        </div>


        {/* RIGHT COLUMN: Canvas, Result, Footer */}
        <div className="right-column">
          <section className="canvas-section-alt">
            <h2 className="canvas-section-title">Canvas</h2>
            <DrawingCanvas 
              selectedStyle={selectedStyle}
              selectedModel={selectedModel}
              styles={styles}
              models={models}
              description={description}
              setDescription={setDescription}
              setSelectedStyle={setSelectedStyle}
              setSelectedModel={setSelectedModel}
              setGenerationResult={setGenerationResult}
              setIsLoading={setIsLoading}
              showNotification={showNotification}
              showCanvasOnly={true}
            />
          </section>

          <section className="result-section">
            <h2 className="result-section-title">Result</h2>
            <div className="result-display">
              <ImageResult 
                generationResult={generationResult} 
                setIsLoading={setIsLoading}
                setGenerationResult={setGenerationResult}
              />
            </div>
          </section>
        </div>

        {isLoading && <LoadingOverlay />}
      </div>
    </ResetProvider>
  );
};

export default App;