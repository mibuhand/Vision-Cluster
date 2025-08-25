import numpy as np
from PIL import Image

def decode_and_save_processed_image(processed_tensor, save_path):
    """
    Decode a processed image tensor and save it as a preview image.
    
    Args:
        processed_tensor: PyTorch tensor of shape [C, H, W] or [B, C, H, W]
        save_path: Path where to save the decoded image
    """
    # Remove batch dimension if present
    if processed_tensor.dim() == 4:
        processed_tensor = processed_tensor[0]
    
    # Convert to numpy and reorder dimensions from CHW to HWC
    processed_numpy = processed_tensor.permute(1, 2, 0).numpy()
    
    # Clip values to valid range and convert to uint8
    processed_numpy = np.clip(processed_numpy, 0, 1)
    preview_image = Image.fromarray((processed_numpy * 255).astype(np.uint8))
    
    # Save the image
    preview_image.save(save_path)
    return str(save_path)