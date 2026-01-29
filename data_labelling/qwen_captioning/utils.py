import os 
import json 
from PIL import Image 
from pathlib import Path

def load_images_from_folder(folder_path, extensions={".jpg", ".jpeg", ".png"}):
    """Loads all image file paths from a given folder.
    Args:
        folder_path (str or Path): Folder containing images
        extensions (set): Allowed image extensions
        
    Returns:
        list of Path: Image file paths
    """
    folder = Path(folder_path)
    return [img_path for img_path in folder.iterdir() if img_path.suffix.lower() in extensions]


def save_jsonl(data, output_path):
    """
    Saves a list of dictionaries to a JSONL file.
    
    Args:
        data (list): List of dicts (image, report)
        output_path (str): Output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log(message):
    """
    Simple Logger.
    """
    print(f"[LOG] {message}")

def load_image(image_path):
    """
    Opens an image using PIL.
    
    Args:
        image_path (str): Path to the image file
        
    Returns: 
        PIL.Image
    """
    return Image.open(image_path).convert("RGB")