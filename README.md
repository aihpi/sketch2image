# Sketch-to-Image Demonstrator

This project is an interactive sketch-to-image demonstrator that allows users to draw freehand sketches and convert them into realistic or artistic images using machine learning.

## Features

- **Intuitive Drawing Interface**: Built with Excalidraw for a natural drawing experience
- **Style Selection**: Choose from multiple visual styles for your generated images
- **Real-time Processing**: Watch as your sketches are transformed into detailed images
- **Responsive Design**: Works on tablets and desktop devices

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
│   ├── package.json         # Frontend dependencies
│   └── Dockerfile           # Frontend Docker configuration
├── backend/                 # FastAPI backend
│   ├── app/                 # Application code
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

2. Configure environment variables (optional):
   - Edit the `.env` file to customize settings like model ID, inference steps, etc.

3. Start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/api

### Using GPU Acceleration

If you have an NVIDIA GPU with CUDA support, uncomment the GPU-related lines in the `docker-compose.yml` file to enable GPU acceleration for the backend service. This will significantly improve the image generation speed.

## Usage Guide

1. **Drawing Your Sketch**:
   - Use the Excalidraw canvas to create your sketch
   - Keep lines clear and distinct for best results
   - Simple sketches often work better than highly detailed ones

2. **Selecting a Style**:
   - Choose from available styles in the dropdown menu
   - Each style produces different artistic results
   - Optionally add a description to guide the generation

3. **Generating Images**:
   - Click "Generate Image" when your sketch is ready
   - Wait 10-30 seconds for the AI to process your sketch
   - When complete, the generated image will appear on the right side

4. **Managing Results**:
   - Download the generated image using the "Download Image" button
   - Clear the canvas to start a new sketch

## Model Information

This project uses the Stable Diffusion model with ControlNet (Scribble) to convert sketches to images. The default model is `lllyasviel/control_v11p_sd15_scribble`, which is well-suited for transforming simple line drawings into detailed images.

You can customize the model by changing the `MODEL_ID` in the `.env` file. Other recommended models include:

- `xinsir/controlnet-scribble-sdxl-1.0` - Higher quality but slower
- `TencentARC/t2i-adapter-sketch-sdxl-1.0` - Good sketch adaptation
- `qninhdt/img2img-turbo` - Fast generation but less detailed

## Development

### Local Development Setup

To run the project without Docker for development:

#### Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend:
```bash
cd frontend
npm install
npm start
```

### API Endpoints

The backend exposes the following API endpoints:

- `GET /api/styles` - Get available style options
- `POST /api/generate` - Generate an image from a sketch
- `GET /api/status/{generation_id}` - Check the status of generation
- `GET /api/images/{generation_id}` - Get the generated image

## Limitations

- Generation time depends on hardware capabilities (10-30 seconds typical)
- Complex sketches may not be interpreted correctly
- The system works best with clear, simple line drawings

## Future Improvements

- Add image upscaling for higher resolution outputs
- Implement user accounts to save and manage generations
- Add more style options and fine-tuned models
- Create a gallery of example sketches and generated images

## License

This project is open source and available under the MIT License.

## Acknowledgements

- [Excalidraw](https://excalidraw.com/) for the drawing interface
- [Hugging Face](https://huggingface.co/) for hosting the pre-trained models
- [ControlNet](https://github.com/lllyasviel/ControlNet) for the sketch-to-image technology