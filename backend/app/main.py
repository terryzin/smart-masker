from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import cv2
import io
import os
import time
import logging
from typing import List, Dict, Optional
from segment_anything import sam_model_registry, SamPredictor

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Request models
class Point(BaseModel):
    x: float
    y: float

class GenerateMaskRequest(BaseModel):
    filename: str
    points: List[Point]

app = FastAPI(title="Smart Masker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'vit_h': {
        'type': 'vit_h',
        'checkpoint': 'sam_vit_h_4b8939.pth',
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    },
    'vit_l': {
        'type': 'vit_l',
        'checkpoint': 'sam_vit_l_0b3195.pth',
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
    },
    'vit_b': {
        'type': 'vit_b',
        'checkpoint': 'sam_vit_b_01ec64.pth',
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    }
}

# Initialize device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Model cache
model_cache = {}
current_model_type = None
predictor = None

def get_model(model_type: str) -> SamPredictor:
    """Get or load a model from cache."""
    global model_cache, current_model_type, predictor
    
    start_time = time.time()
    logger.info(f"Requesting model: {model_type}")
    
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if model_type == current_model_type and predictor is not None:
        logger.info(f"Using cached model: {model_type}")
        return predictor
        
    if model_type not in model_cache:
        config = MODEL_CONFIGS[model_type]
        checkpoint_path = os.path.join(MODEL_DIR, config['checkpoint'])
        
        # Download model if not exists
        if not os.path.exists(checkpoint_path):
            logger.info(f"Downloading {model_type} model checkpoint...")
            download_start = time.time()
            import urllib.request
            urllib.request.urlretrieve(config['url'], checkpoint_path)
            logger.info(f"Model download complete in {time.time() - download_start:.2f}s")
        
        logger.info(f"Loading {model_type} model into memory...")
        load_start = time.time()
        sam = sam_model_registry[config['type']](checkpoint=checkpoint_path)
        sam.to(device=DEVICE)
        model_cache[model_type] = sam
        logger.info(f"Model loaded in {time.time() - load_start:.2f}s")
    
    current_model_type = model_type
    predictor = SamPredictor(model_cache[model_type])
    logger.info(f"Model preparation complete in {time.time() - start_time:.2f}s")
    return predictor

@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    model: str = Form(default='vit_h')
):
    """Upload an image for processing."""
    try:
        total_start = time.time()
        logger.info(f"Starting image upload process for file: {file.filename} with model: {model}")
        
        # Get or load the requested model
        model_start = time.time()
        predictor = get_model(model)
        logger.info(f"Model preparation took {time.time() - model_start:.2f}s")
        
        # Read and validate image
        read_start = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image read complete: size={image.size}, mode={image.mode}, took {time.time() - read_start:.2f}s")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            convert_start = time.time()
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
            logger.info(f"Image conversion took {time.time() - convert_start:.2f}s")
        
        # Save image temporarily
        save_start = time.time()
        filename = os.path.join(UPLOAD_DIR, file.filename)
        image.save(filename)
        logger.info(f"Image saved to: {filename} in {time.time() - save_start:.2f}s")
        
        # Load image for SAM
        preprocess_start = time.time()
        logger.info("Starting SAM image preprocessing...")
        image_array = np.array(image)
        
        # Log memory usage of the image
        image_size_mb = image_array.nbytes / (1024 * 1024)
        logger.info(f"Image array size: {image_size_mb:.2f}MB")
        
        # Set image in predictor
        logger.info("Setting image in SAM predictor...")
        predictor.set_image(image_array)
        logger.info(f"SAM preprocessing complete in {time.time() - preprocess_start:.2f}s")
        
        response_data = {
            "filename": file.filename,
            "size": {"width": image.width, "height": image.height}
        }
        logger.info(f"Upload process complete in {time.time() - total_start:.2f}s")
        return response_data
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate-mask")
async def generate_mask(request: GenerateMaskRequest):
    """Generate mask for given points using SAM2."""
    try:
        total_start = time.time()
        logger.info(f"Starting mask generation for image: {request.filename}")
        logger.info(f"Input points: {request.points}")
        
        image_path = os.path.join(UPLOAD_DIR, request.filename)
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise HTTPException(status_code=404, detail="Image not found")

        # Convert points to numpy arrays
        points_start = time.time()
        input_points = np.array([[p.x, p.y] for p in request.points])
        logger.info(f"Points array shape: {input_points.shape}, conversion took {time.time() - points_start:.2f}s")
        
        # Generate mask
        predict_start = time.time()
        logger.info("Starting SAM prediction...")
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones(len(request.points)),  # All points are foreground
            multimask_output=False  # Get single best mask
        )
        logger.info(f"SAM prediction complete in {time.time() - predict_start:.2f}s")
        logger.info(f"Mask shape: {masks[0].shape}, Prediction score: {scores[0]:.4f}")
        
        # Convert and save mask
        save_start = time.time()
        mask = masks[0].astype(np.uint8) * 255
        mask_image = Image.fromarray(mask)
        
        mask_filename = f"{os.path.splitext(request.filename)[0]}_mask.png"
        mask_path = os.path.join(UPLOAD_DIR, mask_filename)
        mask_image.save(mask_path)
        logger.info(f"Mask saved to: {mask_path} in {time.time() - save_start:.2f}s")
        
        response_data = {"mask_filename": mask_filename}
        logger.info(f"Mask generation complete in {time.time() - total_start:.2f}s")
        return response_data
    except Exception as e:
        logger.error(f"Mask generation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/download-mask/{filename}")
async def download_mask(filename: str):
    """Download the generated mask."""
    try:
        logger.info(f"Download request for mask: {filename}")
        mask_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(mask_path):
            logger.error(f"Mask not found: {mask_path}")
            raise HTTPException(status_code=404, detail="Mask not found")
        logger.info("Sending mask file")
        return FileResponse(mask_path)
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=404, detail="Mask not found")

# Cleanup endpoint (optional, for development)
@app.on_event("shutdown")
async def cleanup():
    """Clean up temporary files on shutdown."""
    import shutil
    if os.path.exists(UPLOAD_DIR):
        logger.info(f"Cleaning up upload directory: {UPLOAD_DIR}")
        shutil.rmtree(UPLOAD_DIR) 