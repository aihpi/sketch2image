import React from 'react';
import { useColorblind } from '../ColorblindContext';

const ColorblindToggle: React.FC = () => {
  const { isColorblindMode, toggleColorblindMode } = useColorblind();

  return (
    <button 
      className="colorblind-toggle"
      onClick={toggleColorblindMode}
      aria-label={isColorblindMode ? "Switch to standard colors" : "Switch to colorblind-friendly colors"}
      title={isColorblindMode ? "Standard colors" : "Colorblind-friendly colors"}
    >
      <svg 
        width="20" 
        height="20" 
        viewBox="0 0 24 24" 
        fill="none" 
        stroke="currentColor" 
        strokeWidth="2" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      >
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
        <circle cx="12" cy="12" r="3"></circle>
      </svg>
      <span className="toggle-text">
        {isColorblindMode ? 'standard' : 'accessible'}
      </span>
    </button>
  );
};

export default ColorblindToggle;