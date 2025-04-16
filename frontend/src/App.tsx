import React, { useState, useEffect } from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './styles/App.css';

// Components
import DrawingCanvas from './components/DrawingCanvas';
import StyleSelector from './components/StyleSelector';
import ImageResult from './components/ImageResult';
import LoadingOverlay from './components/LoadingOverlay';

// Services
import { fetchStyles } from './services/api';

// Types
import { Style, GenerationResult } from './types';

const App: React.FC = () => {
  const [styles, setStyles] = useState<Style[]>([]);
  const [selectedStyle, setSelectedStyle] = useState<Style | null>(null);
  const [description, setDescription] = useState<string>('');
  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Fetch available styles on component mount
  useEffect(() => {
    const getStyles = async () => {
      try {
        const stylesData = await fetchStyles();
        setStyles(stylesData);
        if (stylesData.length > 0) {
          setSelectedStyle(stylesData[0]);
        }
      } catch (error) {
        console.error('Failed to fetch styles:', error);
      }
    };

    getStyles();
  }, []);

  return (
    <div className="app">
      <ToastContainer position="top-right" autoClose={5000} />
      
      <header className="app-header">
        <h1>Sketch to Image</h1>
        <p>Draw a sketch and see it transformed into a realistic image</p>
      </header>

      <main className="app-content">
        <div className="canvas-container">
          <h2>Your Sketch</h2>
          <DrawingCanvas 
            selectedStyle={selectedStyle}
            description={description}
            setDescription={setDescription}
            setGenerationResult={setGenerationResult}
            setIsLoading={setIsLoading}
          />
          <div className="style-controls">
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

      {isLoading && <LoadingOverlay />}
    </div>
  );
};

export default App;