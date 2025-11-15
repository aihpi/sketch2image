import React, { createContext, useContext, useState, useEffect } from 'react';

interface ColorblindContextType {
  isColorblindMode: boolean;
  toggleColorblindMode: () => void;
}

const ColorblindContext = createContext<ColorblindContextType | undefined>(undefined);

export const ColorblindProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isColorblindMode, setIsColorblindMode] = useState(() => {
    const saved = localStorage.getItem('colorblindMode');
    return saved === 'true';
  });

  useEffect(() => {
    if (isColorblindMode) {
      document.documentElement.classList.add('colorblind-mode');
    } else {
      document.documentElement.classList.remove('colorblind-mode');
    }
    localStorage.setItem('colorblindMode', isColorblindMode.toString());
  }, [isColorblindMode]);

  const toggleColorblindMode = () => {
    setIsColorblindMode(prev => !prev);
  };

  return (
    <ColorblindContext.Provider value={{ isColorblindMode, toggleColorblindMode }}>
      {children}
    </ColorblindContext.Provider>
  );
};

export const useColorblind = () => {
  const context = useContext(ColorblindContext);
  if (context === undefined) {
    throw new Error('useColorblind must be used within a ColorblindProvider');
  }
  return context;
};