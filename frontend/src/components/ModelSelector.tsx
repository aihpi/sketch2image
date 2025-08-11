import React from 'react';
import { Model } from '../types';
import Icon from './Icon';

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

  const getModelIconName = (model: Model) => {
    if (model.inference_speed.includes('Fast')) {
      return 'lightning';
    } else if (model.inference_speed.includes('Medium')) {
      return 'target';
    } else {
      return 'diamond';
    }
  };

  return (
    <div className="model-selector">
      <label className="control-label">AI Model</label>
      <div className="model-pills">
        {models.map((model) => (
          <button
            key={model.id}
            className={`model-pill ${selectedModel?.id === model.id ? 'selected' : ''}`}
            onClick={() => setSelectedModel(model)}
            type="button"
          >
            <div className="model-indicator">
              <Icon name={getModelIconName(model)} size={18} />
            </div>
            <div className="model-text">
              <span className="model-name">{model.name}</span>
              <div className="model-tags">
                {model.recommended_for.map((tag, index) => (
                  <span key={index} className="model-tag">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ModelSelector;