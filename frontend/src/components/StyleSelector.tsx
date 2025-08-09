import React from 'react';
import { Style } from '../types';
import Icon from './Icon';
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

  const getStyleIconName = (styleId: string) => {
    switch (styleId) {
      case 'photorealistic':
        return 'camera';
      case 'anime':
        return 'sparkle';
      case 'oil_painting':
        return 'paintbrush';
      case 'watercolor':
        return 'droplet';
      case 'sketch':
        return 'pencil';
      default:
        return 'image';
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
            <div className="style-icon">
              <Icon name={getStyleIconName(style.id)} size={16} />
            </div>
            <span className="style-text">{style.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default StyleSelector;