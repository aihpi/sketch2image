import React, { useState, useEffect } from 'react';
import './styles/App.css';

// Components
import DrawingCanvas from './components/DrawingCanvas';
import StyleSelector from './components/StyleSelector';
import ModelSelector from './components/ModelSelector';
import ImageResult from './components/ImageResult';
import LoadingOverlay from './components/LoadingOverlay';
import Notification from './components/Notification';
import { ResetProvider } from './ResetContext'; // Import the context

// Services
import { fetchStyles, fetchModels } from './services/api';

// Types
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

  // Show notification function
  const showNotification = (message: string, type: 'error' | 'info' | 'success') => {
    setNotification({
      show: true,
      message,
      type
    });
  };

  // Close notification function
  const closeNotification = () => {
    setNotification(prev => ({ ...prev, show: false }));
  };

  // Fetch available styles and models on component mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch styles
        const stylesData = await fetchStyles();
        setStyles(stylesData);
        if (stylesData.length > 0) {
          setSelectedStyle(stylesData[0]);
        }

        // Fetch models
        const modelsData = await fetchModels();
        setModels(modelsData);
        if (modelsData.length > 0) {
          setSelectedModel(modelsData[0]);
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
        
        <header className="app-header">
          <div className="header-content">
            <div className="title-section">
              <h1>Sketch to Image</h1>
              <p>Draw a sketch and see it transformed into a realistic image</p>
            </div>
          </div>
        </header>

        <main className="app-content">
          <div className="canvas-container">
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
            />
            
            <div className="style-controls">
              <ModelSelector 
                models={models} 
                selectedModel={selectedModel} 
                setSelectedModel={setSelectedModel} 
              />
              <StyleSelector 
                styles={styles} 
                selectedStyle={selectedStyle} 
                setSelectedStyle={setSelectedStyle} 
              />
              <div className="description-input">
                <label htmlFor="description">Description</label>
                <input
                  type="text"
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="e.g., a cat sitting on a chair"
                />
              </div>
            </div>
          </div>

          <div className="result-container">
            <ImageResult 
              generationResult={generationResult} 
              setIsLoading={setIsLoading}
              setGenerationResult={setGenerationResult}
            />
          </div>
        </main>

        <footer className="app-footer">
          <div className="footer-content">
            <div className="logos-container">
              <img 
                src="/logos.jpg" 
                alt="KI Service Zentrum by Hasso-Plattner-Institut - Gefördert vom Bundesministerium für Bildung und Forschung" 
                className="footer-logos"
              />
            </div>
            <p className="footer-text">
              Powered by <a href="https://hpi.de/en/research/hpi-data-center/ai-service-center/" className="footer-link">HPI's AI Service Center</a>
            </p>
          </div>
        </footer>

        {isLoading && <LoadingOverlay />}
      </div>
    </ResetProvider>
  );
};

export default App;