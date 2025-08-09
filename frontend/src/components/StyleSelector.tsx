import React from 'react';
import { Style } from '../types';
import '../styles/StyleSelector.css';

interface StyleSelectorProps {
  styles: Style[];
  selectedStyle: Style | null;
  setSelectedStyle: React.Dispatch<React.SetStateAction<Style | null>>;
}

const StyleSelector: React.FC<StyleSelectorProps> = ({
  styles,
  selectedStyle,
  setSelectedStyle,
}) => {
  if (styles.length === 0) {
    return <div className="style-selector loading">Loading styles...</div>;
  }

  const getStyleIcon = (styleId: string) => {
    switch (styleId) {
      case 'photorealistic':
        return '📷';
      case 'anime':
        return '✨';
      case 'oil_painting':
        return '🎨';
      case 'watercolor':
        return '🌊';
      case 'sketch':
        return '✏️';
      default:
        return '🎭';
    }
  };

  return (
    <div className="style-selector">
      <label className="selector-label">Style</label>
      <div className="style-pills">
        {styles.map((style) => (
          <button
            key={style.id}
            className={`style-pill ${selectedStyle?.id === style.id ? 'selected' : ''}`}
            onClick={() => setSelectedStyle(style)}
            type="button"
          >
            <span className="style-emoji">{getStyleIcon(style.id)}</span>
            <span className="style-text">{style.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default StyleSelector;