# Smart Masker

Smart Masker is an interactive image segmentation tool that uses Meta's Segment Anything Model (SAM1 & SAM2) to generate high-quality masks from user clicks. It features a modern web interface for easy interaction and supports multiple SAM model variants.

## Features

- 🖱️ Interactive point-based segmentation
- 🎨 Multiple mask support with different colors
- 🔄 Real-time mask generation
- 🖼️ Image zoom and pan functionality
- 📦 Support for both SAM1 and SAM2 model variants
- 💾 Export combined masks
- 🎯 Precise coordinate display
- 🖥️ Modern, responsive UI

## Prerequisites

- Python 3.8+
- Node.js 14+
- CUDA-capable GPU (recommended)

## Installation

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/terryzin/smart-masker.git
cd smart-masker
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload
```

2. In a separate terminal, start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Select a SAM model variant from the dropdown menu:
   - SAM2 models for best performance (recommended)
   - SAM1 models for legacy support
2. Upload an image using the "Select Image" button or drag and drop
3. Click on objects in the image to generate masks
4. Use mouse wheel to zoom in/out
5. Middle mouse button to pan the image
6. Click on existing masks to remove them
7. Use "Clear All Masks" to remove all masks
8. Export the combined mask using the "Export Mask" button

## Model Information

The application supports both SAM1 and SAM2 model variants:

### SAM2 Models (Recommended)
- ViT-H (2.5GB): Best quality, improved performance
- ViT-L (1.3GB): Balanced performance
- ViT-B (385MB): Fast, good quality

### SAM1 Models (Legacy)
- ViT-H (2.4GB): High quality
- ViT-L (1.2GB): Balanced
- ViT-B (375MB): Fast

Models will be automatically downloaded on first use. SAM2 models generally offer better performance and quality compared to SAM1.

## Technical Details

### Backend
- FastAPI for the REST API
- Segment Anything Model (SAM1 & SAM2) for image segmentation
- CUDA acceleration support
- Efficient image processing with NumPy and OpenCV

### Frontend
- React with TypeScript
- Material-UI components
- Canvas-based image manipulation
- Real-time mask visualization
- Responsive design

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta Research
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2) by Meta Research
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Material-UI](https://mui.com/)