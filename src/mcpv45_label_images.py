#!/usr/bin/env python3
"""
MiniCPM-V-4.5 Image Labeling Script

This script processes images using MiniCPM-V-4.5 model to generate comprehensive labels
across multiple categories including people, scene, objects, location, activities, sentiment, and more.

Usage:
    python mcpv45_label_images.py [--input-dir DIR] [--output-file FILE] [--cache-dir DIR]
"""

import argparse
import json
import logging
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(cache_dir: str = None):
    """Load MiniCPM-V-4.5 model and tokenizer"""
    model_name = "openbmb/MiniCPM-V-4_5"
    trust_remote_code = True
    local_files_only = True
    
    # Set up model loading parameters following official documentation
    model_kwargs = {
        'trust_remote_code': trust_remote_code,
        'local_files_only': local_files_only,
        'attn_implementation': 'sdpa',
        'dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    tokenizer_kwargs = {
        'trust_remote_code': trust_remote_code,
        'local_files_only': local_files_only
    }
    
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    logging.info(f"Loading {model_name}...")
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Move model to GPU and set to eval mode
    if torch.cuda.is_available():
        model = model.eval().cuda()
        logging.info("Using device: cuda")
    else:
        model = model.eval()
        logging.info("Using device: cpu")
    
    return model, tokenizer


def get_image_files(directory: Path) -> list[Path]:
    """
    Find all image files in the specified directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        Sorted list of image file paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def label_image(model, tokenizer, image_path: Path):
    """Generate labels for a single image across multiple categories"""
    image = Image.open(image_path).convert('RGB')
    
    prompt = """Analyze this image and provide labels for the following categories. Return your response in JSON format:

{
  "people": ["list of people descriptions (age group, gender, clothing, poses, etc.)"],
  "scene": "brief description of the overall scene/setting",
  "objects": ["list of prominent objects visible in the image"],
  "place": "most likely location/place type (indoor/outdoor, specific venue type)",
  "activities": ["list of activities or actions happening in the image"],
  "sentiment": "emotional tone or mood of the image (positive, negative, neutral, joyful, etc.)",
  "others": ["any other notable aspects, colors, weather, time of day, etc."]
}

Be specific but concise. If a category doesn't apply, use an empty list [] or "none"."""

    # Format messages according to official documentation
    msgs = [{'role': 'user', 'content': [image, prompt]}]
    
    # Use the chat function with official parameters
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        enable_thinking=False,  # Fast mode for batch processing
        stream=False  # Get complete response at once
    )
    
    try:
        # Extract JSON from response
        if isinstance(answer, str):
            response_text = answer.strip()
        else:
            # Handle streamed response if stream=True was used
            response_text = "".join(answer).strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
        
        labels = json.loads(response_text)
        return labels
    except json.JSONDecodeError:
        # Fallback: return raw response if JSON parsing fails
        logging.warning(f"Failed to parse JSON response for {image_path.name}")
        return {"raw_response": response_text if 'response_text' in locals() else str(answer)}


def process_images_batch(model, tokenizer, image_paths: list[Path], batch_size: int = 1):
    """Process a batch of images and return their labels"""
    batch_labels = {}
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        logging.info(f"Processing batch {batch_num}/{total_batches}: {len(batch_paths)} images")
        
        for image_path in batch_paths:
            try:
                labels = label_image(model, tokenizer, image_path)
                batch_labels[image_path.name] = labels
                logging.info(f"  {image_path.name}: Labels generated successfully")
            except Exception as e:
                logging.error(f"  {image_path.name}: Error - {str(e)}")
                batch_labels[image_path.name] = {"error": str(e)}
    
    return batch_labels


def process_images(input_dir: Path, output_file: Path, cache_dir: str = None, batch_size: int = 32):
    """Process all images in the input directory and save labels"""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Load model
    model, tokenizer = load_model(cache_dir)
    
    # Get all image files
    image_files = get_image_files(input_dir)
    
    if not image_files:
        logging.warning(f"No image files found in {input_dir}")
        return {}
    
    logging.info(f"Processing {len(image_files)} images in batches of {batch_size}...")
    
    # Process images in batches
    image_labels = process_images_batch(model, tokenizer, image_files, batch_size)
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(image_labels, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Labeling complete! Results saved to {output_file}")
    return image_labels


def main():
    """Main execution function."""
    # Directory setup
    proj_dir = Path(__file__).resolve().parent.parent
    
    parser = argparse.ArgumentParser(description='Label images using MiniCPM-V-4.5 model')
    parser.add_argument('--input-dir', type=Path, 
                        default=proj_dir / 'data' / 'processed',
                        help='Input directory containing images to label')
    parser.add_argument('--output-file', type=Path,
                        default=proj_dir / 'data' / 'image_labels.json',
                        help='Output JSON file for labels')
    parser.add_argument('--cache-dir', type=str,
                        default=str(proj_dir / '.huggingface'),
                        help='Cache directory for model files')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing images (default: 32)')
    
    args = parser.parse_args()
    
    try:
        # Process images
        labels = process_images(args.input_dir, args.output_file, args.cache_dir, args.batch_size)
        
        # Print summary
        total_images = len(labels)
        successful_labels = sum(1 for label in labels.values() if "error" not in label)
        
        logging.info(f"Summary:")
        logging.info(f"Total images: {total_images}")
        logging.info(f"Successfully labeled: {successful_labels}")
        logging.info(f"Errors: {total_images - successful_labels}")
        
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())