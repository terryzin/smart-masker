import os
import time
import logging
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SAMHandler:
    def __init__(self, model_dir: str, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        self.model_cache = {}
        self.current_model = None
        self.predictor = None
        self.auto_mask_generator = None
        self.current_sam = None
        
        self.MODEL_CONFIGS = {
            'vit_h': {
                'type': 'vit_h',
                'checkpoint': 'sam_vit_h_4b8939.pth',
            },
            'vit_l': {
                'type': 'vit_l',
                'checkpoint': 'sam_vit_l_0b3195.pth',
            },
            'vit_b': {
                'type': 'vit_b',
                'checkpoint': 'sam_vit_b_01ec64.pth',
            }
        }
    
    def get_model(self, model_type: str) -> SamPredictor:
        """Get or load a SAM model from cache."""
        start_time = time.time()
        logger.info(f"Requesting SAM model: {model_type}")
        
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid SAM model type: {model_type}")
        
        if model_type == self.current_model and self.predictor is not None:
            logger.info(f"Using cached SAM model: {model_type}")
            return self.predictor
            
        if model_type not in self.model_cache:
            config = self.MODEL_CONFIGS[model_type]
            checkpoint_path = os.path.join(self.model_dir, config['checkpoint'])
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"Model file not found: {checkpoint_path}")
                raise RuntimeError(f"Model file not found: {checkpoint_path}")
            
            logger.info(f"Loading {model_type} model into memory...")
            load_start = time.time()
            
            try:
                sam = sam_model_registry[config['type']](checkpoint=checkpoint_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                self.current_sam = sam
                self.auto_mask_generator = None  # Reset auto mask generator when model changes
                logger.info(f"SAM model loaded in {time.time() - load_start:.2f}s")
                
                self.model_cache[model_type] = sam
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Failed to load model: {str(e)}")
                
        self.current_model = model_type
        logger.info(f"SAM model preparation complete in {time.time() - start_time:.2f}s")
        return self.predictor
    
    def set_image(self, image: np.ndarray):
        """Set image for prediction."""
        if self.predictor is None:
            raise RuntimeError("No model loaded")
        self.predictor.set_image(image)
    
    def predict(self, points: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Generate mask for given points."""
        if self.predictor is None:
            raise RuntimeError("No model loaded")
            
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits
        }
    
    def generate_masks(self, image: np.ndarray, **kwargs) -> List[np.ndarray]:
        """Generate automatic masks for the image."""
        if self.current_sam is None:
            raise RuntimeError("No model loaded")
            
        logger.info("Creating automatic mask generator...")
        self.auto_mask_generator = SamAutomaticMaskGenerator(
            model=self.current_sam,
            points_per_side=kwargs.get('points_per_side', 32),
            pred_iou_thresh=kwargs.get('pred_iou_thresh', 0.88),
            stability_score_thresh=kwargs.get('stability_score_thresh', 0.95),
            box_nms_thresh=kwargs.get('box_nms_thresh', 0.7),
            min_mask_region_area=kwargs.get('min_mask_region_area', 0)
        )
        
        logger.info("Generating masks...")
        masks = self.auto_mask_generator.generate(image)
        logger.info(f"Generated {len(masks)} masks")
        
        # Convert masks to numpy arrays
        mask_arrays = []
        for mask in masks:
            mask_array = mask['segmentation'].astype(np.uint8)
            mask_arrays.append(mask_array)
        
        return mask_arrays
    
    def is_sam2(self) -> bool:
        """Check if this is a SAM2 model."""
        return False