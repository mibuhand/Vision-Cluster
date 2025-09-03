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
import re
import subprocess
from datetime import datetime
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


def parse_exif_date(date_string: str) -> Optional[str]:
    """
    Parse EXIF date string and convert to ISO 8601 format.
    
    Args:
        date_string: Raw date string from EXIF data
        
    Returns:
        ISO 8601 formatted date string or None if parsing fails
    """
    if not date_string:
        return None
    
    # Common EXIF date formats
    date_patterns = [
        r'(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})',  # 2023:01:01 12:00:00
        r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})',  # 2023-01-01T12:00:00
        r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})',  # 2023-01-01 12:00:00
    ]
    
    for pattern in date_patterns:
        match = re.match(pattern, date_string.strip())
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                return dt.isoformat()
            except ValueError:
                continue
    
    return None


def parse_gps_coordinate(coord_value, ref_value: str = None) -> Optional[float]:
    """
    Parse GPS coordinate and apply reference direction to get signed decimal degrees.
    
    Args:
        coord_value: GPS coordinate value (could be float, string, or combined string like "32.986369 N")
        ref_value: Reference direction ('N', 'S', 'E', 'W') - optional if included in coord_value
        
    Returns:
        Signed decimal coordinate or None if parsing fails
    """

    if coord_value is None:
        return None
    
    try:
        # Handle combined coordinate strings like "32.986369 N"
        if isinstance(coord_value, str) and ' ' in coord_value:
            parts = coord_value.strip().split()
            if len(parts) == 2:
                coord_float = float(parts[0])
                ref_value = parts[1]  # Override ref_value with the one from the string
            else:
                coord_float = float(coord_value)
        else:
            # Convert to float if it's not already
            coord_float = float(coord_value)
        
        # Apply sign based on reference direction
        if ref_value:
            ref_upper = ref_value.upper()
            if ref_upper in ['S', 'SOUTH', 'W', 'WEST']:
                coord_float = -abs(coord_float)
            elif ref_upper in ['N', 'NORTH', 'E', 'EAST']:
                coord_float = abs(coord_float)
        
        return coord_float
    except (ValueError, TypeError):
        return None


def extract_date_and_gps(metadata: Dict) -> Dict:
    """
    Extract media created date and GPS information from EXIF metadata.
    
    Args:
        metadata: Dictionary containing full EXIF metadata
        
    Returns:
        Dictionary with filtered date and GPS information
    """
    result = {}
    
    # Extract date information (try multiple date fields in order of preference)
    date_fields = [
        'DateTimeOriginal',
        'CreateDate', 
        'DateTime',
    ]
    
    for field in date_fields:
        if field in metadata and metadata[field]:
            parsed_date = parse_exif_date(metadata[field])
            if parsed_date:
                result['created_date'] = parsed_date
                break
    
    # Extract GPS information with consistent formatting
    lat_coord = parse_gps_coordinate(
        metadata.get('GPSLatitude'), 
        metadata.get('GPSLatitudeRef')
    )
    lng_coord = parse_gps_coordinate(
        metadata.get('GPSLongitude'), 
        metadata.get('GPSLongitudeRef')
    )
    
    if lat_coord is not None and lng_coord is not None:
        result['latitude'] = lat_coord
        result['longitude'] = lng_coord
    elif lat_coord is not None or lng_coord is not None:
        # Include partial GPS data if available
        if lat_coord is not None:
            result['latitude'] = lat_coord
        if lng_coord is not None:
            result['longitude'] = lng_coord
    
    return result


def process_and_filter_metadata(results: list) -> list:
    """
    Process full metadata results and extract only date and GPS information.
    
    Args:
        results: List of full metadata results
        
    Returns:
        List of filtered results with only date and GPS data
    """
    filtered_results = []
    
    for result in results:
        if 'metadata' in result:
            date_gps_data = extract_date_and_gps(result['metadata'])
            
            # Only include files that have either date or GPS data
            if date_gps_data:
                filtered_result = {
                    'filename': result['filename'],
                    **date_gps_data
                }
                filtered_results.append(filtered_result)
    
    return filtered_results


def main():
    """Main execution function."""
    # Directory setup - use raw images as default
    proj_dir = Path(__file__).resolve().parent.parent
    raw_dir = proj_dir / 'media' / 'raw'
    
    parser = argparse.ArgumentParser(description='Extract EXIF metadata from image files')
    parser.add_argument('--input-dir', default=str(raw_dir),
                        help=f'Input directory containing image files (default: {raw_dir})')
    parser.add_argument('--output', default=str(proj_dir / 'data' / 'metadata.json'),
                        help='Output JSON file for metadata results')
    parser.add_argument('--date-gps-output', default=str(proj_dir / 'data' / 'date_gps.json'),
                        help='Output JSON file for filtered date and GPS data')
    
    args = parser.parse_args()
    
    # Check if exiftool is available
    if not check_exiftool():
        logging.error("exiftool is not installed or not available in PATH.")
        logging.error("Please install exiftool from https://exiftool.org/")
        return
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    date_gps_file = Path(args.date_gps_output)
    
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
    
    # Process and save filtered date/GPS data
    filtered_results = process_and_filter_metadata(results)
    
    if filtered_results:
        date_gps_file.parent.mkdir(parents=True, exist_ok=True)
        with open(date_gps_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, indent=2, ensure_ascii=False)
        
        # Count files with date and GPS data separately
        files_with_date = sum(1 for r in filtered_results if 'created_date' in r)
        files_with_gps = sum(1 for r in filtered_results if 'latitude' in r and 'longitude' in r)
        files_with_partial_gps = sum(1 for r in filtered_results if ('latitude' in r) != ('longitude' in r))
        
        logging.info(f"Date and GPS data saved to: {date_gps_file}")
        logging.info(f"Files with date/GPS data: {len(filtered_results)}/{len(results)}")
        logging.info(f"Files with date: {files_with_date}, GPS: {files_with_gps}, partial GPS: {files_with_partial_gps}")
    else:
        logging.warning("No files found with date or GPS information")
    
    # Summary statistics
    total_files_checked = len(all_files)
    supported_files = len(results)
    
    logging.info(f"Summary: {supported_files}/{total_files_checked} files supported by exiftool and processed")

if __name__ == "__main__":
    main()