import React from 'react';
import { Model } from '../types';
import '../styles/ModelSelector.css';

interface ModelSelectorProps {
  models: Model[];
  selectedModel: Model | null;
  setSelectedModel: React.Dispatch<React.SetStateAction<Model | null>>;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  setSelectedModel,
}) => {
  if (models.length === 0) {
    return <div className="model-selector loading">Loading models...</div>;
  }

  return (
    <div className="model-selector">
      <label htmlFor="model-select">Select AI Model:</label>
      <select
        id="model-select"
        value={selectedModel?.id || ''}
        onChange={(e) => {
          const selected = models.find(model => model.id === e.target.value) || null;
          setSelectedModel(selected);
        }}
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default ModelSelector;