import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './styles/App.css';

// Components
import DrawingCanvas from './components/DrawingCanvas';
import StyleSelector from './components/StyleSelector';
import ModelSelector from './components/ModelSelector';
import ImageResult from './components/ImageResult';
import LoadingOverlay from './components/LoadingOverlay';

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
        // Only show error toast for critical failures
        toast.error('Failed to connect to the server. Please check your connection and reload.');
      }
    };

    fetchData();
  }, []);

  return (
    <div className="app">
      {/* Keep ToastContainer but limit its use to only critical errors */}
      <ToastContainer position="top-right" autoClose={5000} limit={1} />
      
      <header className="app-header">
        <h1>Sketch to Image</h1>
        <p>Draw a sketch and see it transformed into a realistic image</p>
      </header>

      <main className="app-content">
        <div className="canvas-container">
          <h2>Your Sketch</h2>
          <DrawingCanvas 
            selectedStyle={selectedStyle}
            selectedModel={selectedModel}
            description={description}
            setDescription={setDescription}
            setGenerationResult={setGenerationResult}
            setIsLoading={setIsLoading}
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
              <label htmlFor="description">Optional Description:</label>
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
          <h2>Generated Image</h2>
          <ImageResult 
            generationResult={generationResult} 
            setIsLoading={setIsLoading}
          />
        </div>
      </main>

      {isLoading && <LoadingOverlay modelName={selectedModel?.name} />}
    </div>
  );
};

export default App;