// frontend/src/services/api.ts
import axios from 'axios';
import { Style, Model, GenerationResult } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Fetch available style options from the API
 */
export const fetchStyles = async (): Promise<Style[]> => {
  try {
    const response = await api.get('/styles');
    return response.data;
  } catch (error) {
    console.error('Error fetching styles:', error);
    throw error;
  }
};

/**
 * Fetch available model options from the API
 */
export const fetchModels = async (): Promise<Model[]> => {
  try {
    const response = await api.get('/models');
    return response.data;
  } catch (error) {
    console.error('Error fetching models:', error);
    throw error;
  }
};

/**
 * Upload a sketch and generate an image
 */
export const generateImage = async (
  sketchFile: File,
  styleId: string,
  modelId: string,
  description?: string
): Promise<GenerationResult> => {
  try {
    // Create form data for file upload
    const formData = new FormData();
    formData.append('sketch_file', sketchFile);
    formData.append('style_id', styleId);
    
    if (modelId) {
      formData.append('model_id', modelId);
    }
    
    if (description) {
      formData.append('description', description);
    }

    // Set proper headers for form data
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

/**
 * Check the status of an image generation request
 */
export const checkGenerationStatus = async (generationId: string): Promise<GenerationResult> => {
  try {
    const response = await api.get(`/status/${generationId}`);
    return response.data;
  } catch (error) {
    console.error('Error checking generation status:', error);
    throw error;
  }
};