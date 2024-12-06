import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SAM2Handler:
    def __init__(self, model_dir: str, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        self.model_cache = {}
        self.current_model = None
        self.predictor = None
        
        self.MODEL_CONFIGS = {
            'sam2_large': {
                'type': 'sam2_h',
                'checkpoint': 'sam2.1_hiera_large.pt',
            },
            'sam2_base_plus': {
                'type': 'sam2_l',
                'checkpoint': 'sam2.1_hiera_base_plus.pt',
            },
            'sam2_small': {
                'type': 'sam2_b',
                'checkpoint': 'sam2.1_hiera_small.pt',
            },
            'sam2_tiny': {
                'type': 'sam2_t',
                'checkpoint': 'sam2.1_hiera_tiny.pt',
            }
        }
    
    def get_model(self, model_type: str):
        """Get or load a SAM2 model from cache."""
        start_time = time.time()
        logger.info(f"Requesting SAM2 model: {model_type}")
        
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid SAM2 model type: {model_type}")
        
        if model_type == self.current_model and self.predictor is not None:
            logger.info(f"Using cached SAM2 model: {model_type}")
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
                # Import SAM2 modules here to avoid startup errors if not installed
                try:
                    from segment_anything_2 import build_sam2_model, SamPredictor as Sam2Predictor
                except ImportError:
                    try:
                        from segment_anything_2.modeling import build_sam2_model
                        from segment_anything_2.predictor import SamPredictor as Sam2Predictor
                    except ImportError:
                        raise RuntimeError("SAM2 package not installed")
                
                sam = build_sam2_model(checkpoint_path)
                sam.to(device=self.device)
                self.predictor = Sam2Predictor(sam)
                logger.info(f"SAM2 model loaded in {time.time() - load_start:.2f}s")
                
                self.model_cache[model_type] = sam
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Failed to load model: {str(e)}")
                
        self.current_model = model_type
        logger.info(f"SAM2 model preparation complete in {time.time() - start_time:.2f}s")
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
            multimask_output=False,
            hq_token_only=True
        )
        
        return {
            "masks": masks,
            "scores": scores,
            "logits": logits
        }
    
    def generate_masks(self, image: np.ndarray, **kwargs) -> List[np.ndarray]:
        """Generate automatic masks for the image."""
        if self.predictor is None:
            raise RuntimeError("No model loaded")
            
        masks = self.predictor.generate(
            image,
            points_per_side=kwargs.get('points_per_side', 32),
            pred_iou_thresh=kwargs.get('pred_iou_thresh', 0.88),
            stability_score_thresh=kwargs.get('stability_score_thresh', 0.95),
            box_nms_thresh=kwargs.get('box_nms_thresh', 0.7),
            min_mask_region_area=kwargs.get('min_mask_region_area', 0)
        )
        
        return masks
    
    def is_sam2(self) -> bool:
        """Check if this is a SAM2 model."""
        return True 