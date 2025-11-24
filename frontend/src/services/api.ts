import axios from 'axios';
import { Style, Model, GenerationResult, EnhancedPromptResult } from '../types';

// Use relative path - proxy will handle routing
const API_URL = '/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchStyles = async (): Promise<Style[]> => {
  try {
    const response = await api.get('/styles');
    return response.data;
  } catch (error) {
    console.error('Error fetching styles:', error);
    throw error;
  }
};

export const fetchModels = async (): Promise<Model[]> => {
  try {
    const response = await api.get('/models');
    return response.data;
  } catch (error) {
    console.error('Error fetching models:', error);
    throw error;
  }
};

export const enhancePrompt = async (
  sketchFile: File,
  description: string
): Promise<EnhancedPromptResult> => {
  try {
    const formData = new FormData();
    formData.append('sketch_file', sketchFile);
    formData.append('description', description);

    const response = await api.post('/enhance-prompt', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('Error enhancing prompt:', error);
    throw error;
  }
};

export const generateImage = async (
  sketchFile: File,
  styleId: string,
  modelId: string,
  description?: string
): Promise<GenerationResult> => {
  try {
    const formData = new FormData();
    formData.append('sketch_file', sketchFile);
    formData.append('style_id', styleId);
    
    if (modelId) {
      formData.append('model_id', modelId);
    }
    
    if (description) {
      formData.append('description', description);
    }

    const response = await api.post('/generate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('Error generating image:', error);
    throw error;
  }
};

export const checkGenerationStatus = async (generationId: string): Promise<GenerationResult> => {
  try {
    const response = await api.get(`/status/${generationId}`);
    return response.data;
  } catch (error) {
    console.error('Error checking generation status:', error);
    throw error;
  }
};