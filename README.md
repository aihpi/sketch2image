# Sketch-to-Image Demonstrator

This project is an interactive sketch-to-image demonstrator that allows users to draw freehand sketches and convert them into realistic or artistic images using machine learning.

## Features

- **Intuitive Drawing Interface**: Built with Excalidraw for a natural drawing experience
- **Style Selection**: Choose from multiple visual styles for your generated images (Photorealistic, Anime, Oil Painting, Watercolor, and Detailed Sketch)
- **Responsive Design**: Works on tablets and desktop devices
- **GPU Acceleration**: Utilizes NVIDIA GPUs when available for faster image generation

## Technology Stack

- **Frontend**: React with TypeScript, Excalidraw for drawing
- **Backend**: Python with FastAPI
- **ML Integration**: Hugging Face Diffusers with ControlNet for sketch-to-image conversion
- **Deployment**: Docker for containerization

## Project Structure

```
sketch2image/
├── frontend/                # React frontend application
│   ├── public/              # Static files
│   ├── src/                 # Source code
│   │   ├── components/      # React components
│   │   ├── services/        # API service functions
│   │   ├── styles/          # CSS styles
│   │   └── types.ts         # TypeScript type definitions
│   ├── package.json         # Frontend dependencies
│   └── Dockerfile           # Frontend Docker configuration
├── backend/                 # FastAPI backend
│   ├── app/                 # Application code
│   │   ├── api/             # API endpoints
│   │   ├── core/            # Core functionality
│   │   ├── models/          # Data models
│   │   └── services/        # ML services
│   ├── main.py              # Main application file
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Backend Docker configuration
├── .env                     # Environment variables
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- For better performance: NVIDIA GPU with CUDA support

### Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sketch2image.git
   cd sketch2image
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/api

### GPU Acceleration

The application is configured to automatically use GPU acceleration if an NVIDIA GPU is available. The setup script detects your GPU and configures the application accordingly.

## Usage Guide

1. **Drawing Your Sketch**:
   - Use the Excalidraw canvas to create your sketch
   - Keep lines clear and distinct for best results
   - Simple sketches often work better than highly detailed ones

2. **Selecting a Style**:
   - Choose from available styles in the dropdown menu (Photorealistic, Anime, Oil Painting, etc.)
   - Optionally add a description to guide the generation

3. **Generating Images**:
   - Click "Generate Image" when your sketch is ready
   - Wait for the AI to process your sketch (typically 5-30 seconds with GPU, longer with CPU)
   - When complete, the generated image will appear on the right side

4. **Managing Results**:
   - Download the generated image using the "Download Image" button
   - Clear the canvas to start a new sketch

## Development

### API Endpoints

The backend exposes the following API endpoints:

- `GET /api/styles` - Get available style options
- `POST /api/generate` - Generate an image from a sketch
- `GET /api/status/{generation_id}` - Check the status of generation
- `GET /api/images/{generation_id}` - Get the generated image

## Configuration

The application can be configured through environment variables in the `.env` file:

- `MODEL_ID` - The Hugging Face model ID (default: "lllyasviel/control_v11p_sd15_scribble")
- `NUM_INFERENCE_STEPS` - Number of diffusion steps (default: 20)
- `GUIDANCE_SCALE` - Guidance scale for generation (default: 7.5)
- `OUTPUT_IMAGE_SIZE` - Size of the output image (default: 512)
- `DEVICE` - Device to run inference on ("cuda" or "cpu")

## Limitations

- Generation time depends on hardware capabilities (5-30 seconds on GPU, minutes on CPU)
- Complex sketches may not be interpreted correctly
- The system works best with clear, simple line drawings

## License

This project is open source and available under the MIT License.

## Acknowledgements

- [Excalidraw](https://excalidraw.com/) for the drawing interface
- [Hugging Face](https://huggingface.co/) for hosting the pre-trained models
- [ControlNet](https://github.com/lllyasviel/ControlNet) for the sketch-to-image technology