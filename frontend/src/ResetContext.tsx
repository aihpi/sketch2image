import React, { createContext, useContext, useState } from 'react';

interface ResetContextType {
  resetTrigger: number;
  triggerReset: () => void;
}

const ResetContext = createContext<ResetContextType | undefined>(undefined);

export const ResetProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [resetTrigger, setResetTrigger] = useState(0);

  const triggerReset = () => {
    setResetTrigger(prev => prev + 1);
  };

  return (
    <ResetContext.Provider value={{ resetTrigger, triggerReset }}>
      {children}
    </ResetContext.Provider>
  );
};

export const useReset = () => {
  const context = useContext(ResetContext);
  if (context === undefined) {
    throw new Error('useReset must be used within a ResetProvider');
  }
  return context;
};