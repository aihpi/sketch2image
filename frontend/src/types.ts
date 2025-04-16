export interface Style {
    id: string;
    name: string;
    prompt_prefix?: string;
  }
  
  export interface GenerationResult {
    generation_id: string;
    status: 'processing' | 'completed';
    message: string;
    image_url?: string;
  }