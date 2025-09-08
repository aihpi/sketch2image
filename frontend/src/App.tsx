// src/App.tsx
import React, { useState, useEffect } from 'react';
import './styles/App.css'; // This will be our single source of truth for styles

import DrawingCanvas from './components/DrawingCanvas';
import StyleSelector from './components/StyleSelector';
import ModelSelector from './components/ModelSelector';
import ImageResult from './components/ImageResult';
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
        
        {/* MOBILE HEADER - Only shown on mobile */}
        <header className="app-header mobile-only">
          <h1>Sketch to Image</h1>
        </header>

        {/* MAIN CONTENT WRAPPER */}
        <div className="main-content">
          {/* LEFT COLUMN: Header, Controls & Footer (Desktop) */}
          <div className="controls-column">
            {/* DESKTOP HEADER - Only shown on desktop */}
            <header className="app-header desktop-only">
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

            {/* DESKTOP FOOTER - Only shown on desktop */}
            <footer className="app-footer desktop-only">
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

          {/* RIGHT COLUMN: Canvas & Result (Desktop) / FIRST ON MOBILE */}
          <div className="canvas-result-column">
            <section className="canvas-section">
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

            {/* DESKTOP RESULT - Only shown on desktop */}
            <section className="result-section desktop-only">
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

          {/* MOBILE CONTROLS - Only shown on mobile, comes after canvas */}
          <div className="mobile-controls mobile-only">
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
                <label className="control-label" htmlFor="description-mobile">
                  Description
                </label>
                <textarea
                  id="description-mobile"
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
          </div>

          {/* MOBILE RESULT - Only shown on mobile, comes after controls */}
          <div className="mobile-result mobile-only">
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
        </div>

        {/* MOBILE FOOTER - Only shown on mobile */}
        <footer className="app-footer mobile-only">
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

        {/* Legacy loading overlay - only for backward compatibility if needed */}
        {isLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p className="loading-text">Generating your image...</p>
            <p className="loading-subtext">This may take 10-30 seconds</p>
          </div>
        )}
      </div>
    </ResetProvider>
  );
};

export default App;