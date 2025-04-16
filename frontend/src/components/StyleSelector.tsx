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

  return (
    <div className="style-selector">
      <label htmlFor="style-select">Choose an output style:</label>
      <select
        id="style-select"
        value={selectedStyle?.id || ''}
        onChange={(e) => {
          const selected = styles.find(style => style.id === e.target.value) || null;
          setSelectedStyle(selected);
        }}
      >
        {styles.map((style) => (
          <option key={style.id} value={style.id}>
            {style.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default StyleSelector;