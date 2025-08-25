"""
DINOv3 Image Embedding Extractor

This script processes all images in the data/processed directory and extracts
CLS token embeddings using Facebook's DINOv3 model. The embeddings are saved
as a compressed NumPy archive for downstream tasks like clustering or similarity search.

Features:
- Batch processing for efficient GPU utilization
- Support for multiple image formats (jpg, jpeg, png, bmp, tiff)
- Automatic GPU detection and usage
- Compressed output format (.npz)
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from pathlib import Path
import numpy as np
import glob
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory setup
proj_dir = Path(__file__).resolve().parent.parent
processed_dir = proj_dir / 'data' / 'processed'

def get_image_files(directory: Path) -> List[Path]:
    """
    Find all image files in the specified directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        Sorted list of image file paths
    """
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = set()
    
    for pattern in image_patterns:
        # Search for both lowercase and uppercase extensions
        image_files.update(glob.glob(str(directory / pattern)))
        image_files.update(glob.glob(str(directory / pattern.upper())))
    
    return [Path(f) for f in sorted(image_files)]

# Model configuration
MODEL_NAME = "facebook/dinov3-vit7b16-pretrain-lvd1689m"  # Large DINOv3 model
CACHE_DIR = str(proj_dir / '.huggingface')  # Local cache directory
TRUST_REMOTE_CODE = False  # Security setting
HF_TOKEN = ''  # Hugging Face token (empty for public models)
MAX_BATCH_SIZE = 32  # Maximum batch size for inference to prevent OOM

def load_dinov3_model() -> Tuple[AutoImageProcessor, AutoModel, torch.device]:
    """
    Load DINOv3 model and processor, move model to appropriate device.
    
    Returns:
        Tuple of (processor, model, device)
    """
    # Load image processor
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=TRUST_REMOTE_CODE,
        token=HF_TOKEN,
    )

    # Load pre-trained model
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=TRUST_REMOTE_CODE,
        token=HF_TOKEN,
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"Using device: {device}")
    
    return processor, model, device

def extract_embeddings(image_files: List[Path]) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract CLS token embeddings from images using batched inference.
    
    Args:
        image_files: List of paths to image files
        
    Returns:
        Tuple of (embeddings tensor, image names list)
    """
    # Load model components
    processor, model, device = load_dinov3_model()
    patch_size = model.config.patch_size

    all_embeddings = []
    all_image_names = []
    
    # Process images in batches to prevent OOM
    for i in range(0, len(image_files), MAX_BATCH_SIZE):
        batch_files = image_files[i:i + MAX_BATCH_SIZE]
        batch_size = len(batch_files)
        
        # Load batch of images
        images = []
        image_names = []
        
        for image_file in batch_files:
            image = load_image(str(image_file))
            images.append(image)
            image_names.append(image_file.name)

        # Process batch
        logging.info(f"Processing batch {i//MAX_BATCH_SIZE + 1}: {batch_size} images...")
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to GPU

        # Calculate expected tensor dimensions for validation
        actual_batch_size, _, img_height, img_width = inputs['pixel_values'].shape
        num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
        num_patches_flat = num_patches_height * num_patches_width

        # Run inference
        with torch.inference_mode():
            outputs = model(**inputs)

        # Validate output shape
        last_hidden_states = outputs.last_hidden_state
        expected_shape = (actual_batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)
        assert last_hidden_states.shape == expected_shape, f"Unexpected shape: {last_hidden_states.shape} vs {expected_shape}"

        # Extract CLS tokens (first token in sequence)
        cls_tokens_batch = last_hidden_states[:, 0, :].cpu()  # Move to CPU for storage
        
        all_embeddings.append(cls_tokens_batch)
        all_image_names.extend(image_names)
    
    # Concatenate all batches
    cls_tokens_tensor = torch.cat(all_embeddings, dim=0)
    
    return cls_tokens_tensor, all_image_names

def main():
    """Main execution function."""
    # Get all image files
    image_files = get_image_files(processed_dir)
    
    if not image_files:
        logging.warning(f"No image files found in {processed_dir}")
        return
    
    # Extract embeddings
    cls_tokens_tensor, image_names = extract_embeddings(image_files)

    # Save embeddings and metadata
    output_path = proj_dir / 'data' / 'cls_tokens.npz'
    np.savez(output_path, 
             cls_tokens=cls_tokens_tensor.numpy(), 
             image_names=image_names)
    logging.info(f"CLS tokens saved to: {output_path}")

    # Log summary
    logging.info(f"Processed {len(image_files)} images, embedding dimension: {cls_tokens_tensor.shape[1]}")

if __name__ == "__main__":
    main()