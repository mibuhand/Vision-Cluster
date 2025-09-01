#!/usr/bin/env python3
"""
EXIF Metadata Extractor

This script extracts full EXIF metadata from images using exiftool.
Uses exiftool to read metadata from various image formats including RAW files.

Usage:
    python extract_metadata.py [--input-dir DIR] [--output OUTPUT_FILE]

Requirements:
    - exiftool must be installed and available in PATH
    - pip install (no additional Python packages required)
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_files_in_directory(directory: Path) -> list:
    """
    Find all files supported by exiftool in the directory and extract metadata.
    
    Args:
        directory: Path to directory containing files
        
    Returns:
        List of metadata results with full EXIF data
    """
    results = []
    
    # Get all files in directory (not just known image extensions)
    all_files = [f for f in directory.iterdir() if f.is_file()]
    all_files = sorted(all_files)
    
    if not all_files:
        logging.warning(f"No files found in {directory}")
        return results
    
    for file_path in all_files:
        # Check if file is supported by exiftool and extract metadata
        metadata = run_exiftool(file_path)
        if metadata:
            # Format result with filename and full metadata
            result = {
                'filename': file_path.name,
                'metadata': metadata
            }
            
            results.append(result)
        else:
            logging.warning(f"File not supported by exiftool: {file_path.name}")
    
    return results

def check_exiftool() -> bool:
    """
    Check if exiftool is available in the system PATH.
    
    Returns:
        True if exiftool is available, False otherwise
    """
    try:
        result = subprocess.run(['exiftool', '-ver'], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

def run_exiftool(file_path: Path) -> Optional[Dict]:
    """
    Run exiftool on a file and return the metadata as a dictionary.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary of EXIF metadata or None if extraction fails
    """
    try:
        # Run exiftool with JSON output
        result = subprocess.run(
            ['exiftool', '-json', '-coordFormat', '%.6f', str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout:
            # Parse JSON output (exiftool returns an array with one object)
            metadata_list = json.loads(result.stdout)
            if metadata_list and len(metadata_list) > 0:
                return metadata_list[0]
        
        return None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        logging.debug(f"Exiftool failed for {file_path.name}: {e}")
        return None


def main():
    """Main execution function."""
    # Directory setup - use raw images as default
    proj_dir = Path(__file__).resolve().parent.parent
    raw_dir = proj_dir / 'data' / 'raw'
    
    parser = argparse.ArgumentParser(description='Extract EXIF metadata from image files')
    parser.add_argument('--input-dir', default=str(raw_dir),
                        help=f'Input directory containing image files (default: {raw_dir})')
    parser.add_argument('--output', default=str(proj_dir / 'data' / 'metadata.json'),
                        help='Output JSON file for metadata results')
    
    args = parser.parse_args()
    
    # Check if exiftool is available
    if not check_exiftool():
        logging.error("exiftool is not installed or not available in PATH.")
        logging.error("Please install exiftool from https://exiftool.org/")
        return
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Process all files and extract metadata from supported ones
    all_files = [f for f in input_dir.iterdir() if f.is_file()]
    results = process_files_in_directory(input_dir)
    
    if not results:
        logging.warning(f"No files supported by exiftool found in {input_dir}")
        return
    
    # Write all results to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Metadata extraction complete! Results saved to: {output_file}")
    
    # Summary statistics
    total_files_checked = len(all_files)
    supported_files = len(results)
    
    logging.info(f"Summary: {supported_files}/{total_files_checked} files supported by exiftool and processed")

if __name__ == "__main__":
    main()