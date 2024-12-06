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
from typing import List, Dict, Optional, Union

from .sam_handler import SAMHandler
from .sam2_handler import SAM2Handler

# Configure logging
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

class AutoMaskRequest(BaseModel):
    """Request model for automatic mask generation."""
    filename: str
    points_per_side: Optional[int] = 32
    pred_iou_thresh: Optional[float] = 0.88
    stability_score_thresh: Optional[float] = 0.95
    box_nms_thresh: Optional[float] = 0.7
    min_mask_region_area: Optional[int] = 0

app = FastAPI(title="Smart Masker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Initialize handlers
sam_handler = SAMHandler(MODEL_DIR, DEVICE)
sam2_handler = SAM2Handler(MODEL_DIR, DEVICE)

@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    model: str = Form(default='vit_h')
):
    """Upload an image for processing."""
    try:
        total_start = time.time()
        logger.info(f"Starting image upload process for file: {file.filename} with model: {model}")
        
        if not file.content_type.startswith('image/'):
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Get or load the requested model
        model_start = time.time()
        try:
            if model in sam_handler.MODEL_CONFIGS:
                predictor = sam_handler.get_model(model)
                handler = sam_handler
            elif model in sam2_handler.MODEL_CONFIGS:
                predictor = sam2_handler.get_model(model)
                handler = sam2_handler
            else:
                raise ValueError(f"Invalid model type: {model}")
            logger.info(f"Model preparation took {time.time() - model_start:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        # Read and validate image
        read_start = time.time()
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Image read complete: size={image.size}, mode={image.mode}, took {time.time() - read_start:.2f}s")
        except Exception as e:
            logger.error(f"Failed to read image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            convert_start = time.time()
            logger.info(f"Converting image from {image.mode} to RGB")
            try:
                image = image.convert('RGB')
                logger.info(f"Image conversion took {time.time() - convert_start:.2f}s")
            except Exception as e:
                logger.error(f"Failed to convert image to RGB: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to convert image: {str(e)}")
        
        # Save image temporarily
        save_start = time.time()
        try:
            file_extension = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{int(time.time())}_{os.urandom(4).hex()}{file_extension}"
            filename = os.path.join(UPLOAD_DIR, unique_filename)
            image.save(filename)
            logger.info(f"Image saved to: {filename} in {time.time() - save_start:.2f}s")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")
        
        # Load image for SAM/SAM2
        preprocess_start = time.time()
        logger.info("Starting image preprocessing...")
        try:
            image_array = np.array(image)
            image_size_mb = image_array.nbytes / (1024 * 1024)
            logger.info(f"Image array size: {image_size_mb:.2f}MB")
            
            handler.set_image(image_array)
            logger.info(f"Image preprocessing complete in {time.time() - preprocess_start:.2f}s")
        except Exception as e:
            logger.error(f"Failed in preprocessing: {str(e)}")
            try:
                os.remove(filename)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Failed in image preprocessing: {str(e)}")
        
        response_data = {
            "filename": unique_filename,
            "size": {"width": image.width, "height": image.height}
        }
        logger.info(f"Upload process complete in {time.time() - total_start:.2f}s")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-mask")
async def generate_mask(request: GenerateMaskRequest):
    """Generate mask for given points using SAM/SAM2."""
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
        logger.info("Starting prediction...")
        
        # Get current handler
        if sam_handler.predictor is not None:
            handler = sam_handler
        elif sam2_handler.predictor is not None:
            handler = sam2_handler
        else:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        result = handler.predict(
            points=input_points,
            labels=np.ones(len(request.points))
        )
        
        logger.info(f"Prediction complete in {time.time() - predict_start:.2f}s")
        logger.info(f"Mask shape: {result['masks'][0].shape}, Prediction score: {result['scores'][0]:.4f}")
        
        # Convert and save mask
        save_start = time.time()
        mask = result['masks'][0].astype(np.uint8) * 255
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

@app.post("/api/generate-auto-masks")
async def generate_auto_masks(request: AutoMaskRequest):
    """Generate masks automatically using SAM/SAM2's automatic mask generation."""
    try:
        total_start = time.time()
        logger.info(f"Starting automatic mask generation for image: {request.filename}")
        
        # Get current handler
        if sam_handler.predictor is not None:
            handler = sam_handler
            logger.info("Using SAM handler")
        elif sam2_handler.predictor is not None:
            handler = sam2_handler
            logger.info("Using SAM2 handler")
        else:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        image_path = os.path.join(UPLOAD_DIR, request.filename)
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise HTTPException(status_code=404, detail="Image not found")

        # Read image
        logger.info("Reading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Image loaded: shape={image.shape}")
        
        # Generate masks
        predict_start = time.time()
        logger.info("Starting automatic mask generation...")
        
        try:
            masks = handler.generate_masks(
                image,
                points_per_side=request.points_per_side,
                pred_iou_thresh=request.pred_iou_thresh,
                stability_score_thresh=request.stability_score_thresh,
                box_nms_thresh=request.box_nms_thresh,
                min_mask_region_area=request.min_mask_region_area
            )
            logger.info(f"Generated {len(masks)} masks")
            logger.info(f"Automatic mask generation complete in {time.time() - predict_start:.2f}s")
        except Exception as e:
            logger.error(f"Failed to generate masks: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate masks: {str(e)}")
        
        # Save masks
        save_start = time.time()
        mask_filenames = []
        
        logger.info("Saving masks...")
        for idx, mask in enumerate(masks):
            try:
                mask_array = mask.astype(np.uint8) * 255
                mask_image = Image.fromarray(mask_array)
                
                mask_filename = f"{os.path.splitext(request.filename)[0]}_auto_mask_{idx}.png"
                mask_path = os.path.join(UPLOAD_DIR, mask_filename)
                mask_image.save(mask_path)
                mask_filenames.append(mask_filename)
                logger.info(f"Saved mask {idx + 1}/{len(masks)}: {mask_filename}")
            except Exception as e:
                logger.error(f"Failed to save mask {idx}: {str(e)}")
            
        logger.info(f"Saved {len(mask_filenames)} masks in {time.time() - save_start:.2f}s")
        
        response_data = {
            "mask_filenames": mask_filenames,
            "num_masks": len(mask_filenames)
        }
        logger.info(f"Automatic mask generation complete in {time.time() - total_start:.2f}s")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Automatic mask generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup endpoint (optional, for development)
@app.on_event("shutdown")
async def cleanup():
    """Clean up temporary files on shutdown."""
    import shutil
    if os.path.exists(UPLOAD_DIR):
        logger.info(f"Cleaning up upload directory: {UPLOAD_DIR}")
        shutil.rmtree(UPLOAD_DIR) 