
#!/usr/bin/env python3

import os
import sys
import argparse
from PIL import Image

def tif_to_jpg(tif_path):
    """
    Convert a .tif image to .jpg format and save it in the same directory
    
    Args:
        tif_path (str): Path to the .tif image
    
    Returns:
        str: Path to the saved .jpg image or None if conversion failed
    """
    # Check if the input file exists
    if not os.path.exists(tif_path):
        print(f"Error: File {tif_path} does not exist")
        return None
    
    # Check if the input file is a .tif file
    if not tif_path.lower().endswith(('.tif', '.tiff')):
        print(f"Error: File {tif_path} is not a TIF image")
        return None
    
    try:
        # Open the TIF image
        img = Image.open(tif_path)
        
        # Generate the output path by replacing the extension
        base_path = os.path.splitext(tif_path)[0]
        jpg_path = f"{base_path}.jpg"
        
        # Convert and save as JPG
        if img.mode in ['RGBA', 'LA']:
            # If the image has an alpha channel, convert to RGB
            img = img.convert('RGB')
        
        img.save(jpg_path, "JPEG", quality=95)
        print(f"Successfully converted {tif_path} to {jpg_path}")
        
        return jpg_path
    
    except Exception as e:
        print(f"Error converting {tif_path} to JPG: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert TIF images to JPG format")
    parser.add_argument('tif_path', help="Path to the TIF image file to convert")
    
    args = parser.parse_args()
    tif_to_jpg(args.tif_path)

if __name__ == "__main__":
    main()
