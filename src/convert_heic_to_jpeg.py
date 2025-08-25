#!/usr/bin/env python3
"""
Simple script to convert HEIC photos to JPEG format using ImageMagick
Works on Windows, macOS, and Linux

Usage:
    python convert_heic_to_jpeg.py [input_folder] [output_folder] [--max-size SIZE] [--quality Q]

Requirements:
    - ImageMagick installed and available in PATH
    - Python 3.6+
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def convert_heic_to_jpeg(input_folder, output_folder, max_size=512, quality=85):
    """Convert all HEIC files in input_folder to JPEG in output_folder"""
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all HEIC files (case insensitive)
    heic_files = list(input_path.glob('*.heic')) + list(input_path.glob('*.HEIC'))
    # Remove duplicates that may occur on case-insensitive file systems
    heic_files = list(set(heic_files))
    
    if not heic_files:
        print(f"No HEIC files found in {input_folder}")
        return
    
    print(f"Found {len(heic_files)} HEIC files to convert")
    print(f"Converting from: {input_path.absolute()}")
    print(f"Converting to: {output_path.absolute()}")
    print("-" * 50)
    
    converted = 0
    failed = 0
    
    for heic_file in heic_files:
        try:
            # Create output filename with .jpg extension
            output_file = output_path / f"{heic_file.stem}.jpg"
            
            print(f"Converting: {heic_file.name}")
            
            # Build ImageMagick command
            cmd = ['magick', str(heic_file)]
            
            # Add downsampling if max_size is specified and > 0
            if max_size > 0:
                # Get image dimensions first
                identify_result = subprocess.run([
                    'magick', 'identify', '-format', '%w %h', str(heic_file)
                ], capture_output=True, text=True)
                
                if identify_result.returncode == 0:
                    try:
                        width, height = map(int, identify_result.stdout.strip().split())
                        max_dimension = max(width, height)
                        
                        # Only downsample if the image is larger than max_size
                        if max_dimension > max_size:
                            cmd.extend(['-resize', f'{max_size}x{max_size}>'])
                            print(f"  Downsampling from {width}x{height} (max dimension: {max_dimension})")
                        else:
                            print(f"  Keeping original size {width}x{height} (already small enough)")
                    except (ValueError, IndexError):
                        print(f"  Warning: Could not determine image size, converting without resizing")
            
            # Add quality setting and output file
            cmd.extend(['-quality', str(quality), str(output_file)])
            
            # Run ImageMagick convert command
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
            print(f"  ✗ Error converting {heic_file.name}: {e}")
            failed += 1
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"Successfully converted: {converted} files")
    if failed > 0:
        print(f"Failed to convert: {failed} files")


def main():
    parser = argparse.ArgumentParser(description='Convert HEIC photos to JPEG format using ImageMagick')
    parser.add_argument('input_folder', nargs='?', default='.', 
                        help='Input folder containing HEIC files (default: current directory)')
    parser.add_argument('output_folder', nargs='?', default='converted',
                        help='Output folder for JPEG files (default: ./converted)')
    parser.add_argument('--max-size', type=int, default=512,
                        help='Maximum width or height in pixels (default: 512, 0 = no downsampling)')
    parser.add_argument('--quality', type=int, default=85,
                        help='JPEG quality (1-100, default: 85)')
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    convert_heic_to_jpeg(args.input_folder, args.output_folder, args.max_size, args.quality)


if __name__ == '__main__':
    main()