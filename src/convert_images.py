#!/usr/bin/env python3
"""
Simple script to convert images to JPEG format using ImageMagick
Works on Windows, macOS, and Linux

Usage:
    python convert_images.py [input_folder] [output_folder] [--max-size SIZE] [--quality Q]

Requirements:
    - ImageMagick installed and available in PATH
    - Python 3.6+
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def convert_images_to_jpeg(input_folder, output_folder, max_size=512, quality=85):
    """Convert all image files in input_folder to JPEG in output_folder"""
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files by checking if ImageMagick can identify them
    # Exclude animated/multi-frame images
    image_files = []
    for file_path in input_path.iterdir():
        if file_path.is_file():
            # Check if ImageMagick can identify the file format and frame count
            try:
                identify_result = subprocess.run([
                    'magick', 'identify', '-format', '%m %n', str(file_path)
                ], capture_output=True, text=True, timeout=10)
                
                if identify_result.returncode == 0:
                    format_info = identify_result.stdout.strip().split()
                    if len(format_info) >= 2:
                        format_type = format_info[0]
                        frame_count = int(format_info[1])
                        
                        # Only include single-frame images
                        if frame_count == 1:
                            image_files.append(file_path)
                        else:
                            print(f"Skipping animated image: {file_path.name} ({frame_count} frames)")
                    else:
                        # Fallback: if we can't determine frame count, include it
                        image_files.append(file_path)
            except (subprocess.TimeoutExpired, ValueError, Exception):
                # Skip files that can't be identified or take too long
                continue
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files to convert")
    print(f"Converting from: {input_path.absolute()}")
    print(f"Converting to: {output_path.absolute()}")
    print("-" * 50)
    
    converted = 0
    failed = 0
    
    for image_file in image_files:
        try:
            # Create output filename with .jpg extension
            output_file = output_path / f"{image_file.stem}.jpg"
            
            print(f"Converting: {image_file.name}")
            
            # Build single ImageMagick command with all operations
            cmd = ['magick', str(image_file)]
            
            
            # Add downsampling if max_size is specified and > 0
            if max_size > 0:
                cmd.extend(['-resize', f'{max_size}x{max_size}>'])
            
            # Preserve EXIF metadata and handle orientation
            
            # Add quality setting and output file
            cmd.extend(['-quality', str(quality), str(output_file)])
            
            # Run single ImageMagick command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✓ Success: {output_file.name}")
                converted += 1
            else:
                print(f"  ✗ Error: {result.stderr.strip()}")
                failed += 1
                
        except FileNotFoundError:
            print("Error: ImageMagick not found. Please install ImageMagick and ensure it's in your PATH.")
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ Error converting {image_file.name}: {e}")
            failed += 1
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"Successfully converted: {converted} files")
    if failed > 0:
        print(f"Failed to convert: {failed} files")


def main():
    # Directory setup similar to dinov3_embeddings.py
    proj_dir = Path(__file__).resolve().parent.parent
    raw_dir = proj_dir / 'media' / 'raw'
    processed_dir = proj_dir / 'media' / 'processed'
    
    parser = argparse.ArgumentParser(description='Convert images to JPEG format using ImageMagick')
    parser.add_argument('input_folder', nargs='?', default=str(raw_dir), 
                        help=f'Input folder containing image files (default: {raw_dir})')
    parser.add_argument('output_folder', nargs='?', default=str(processed_dir),
                        help=f'Output folder for JPEG files (default: {processed_dir})')
    parser.add_argument('--max-size', type=int, default=512,
                        help='Maximum width or height in pixels (default: 512, 0 = no downsampling)')
    parser.add_argument('--quality', type=int, default=85,
                        help='JPEG quality (1-100, default: 85)')
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    convert_images_to_jpeg(args.input_folder, args.output_folder, args.max_size, args.quality)


if __name__ == '__main__':
    main()