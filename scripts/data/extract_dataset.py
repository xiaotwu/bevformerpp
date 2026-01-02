#!/usr/bin/env python3
"""
Dataset extraction and verification script for nuScenes v1.0-mini.
Extracts v1.0-mini.tgz to data directory and verifies dataset structure.
"""

import os
import tarfile
import json
from pathlib import Path
from typing import List, Dict


def extract_dataset(archive_path: str = "v1.0-mini.tgz", 
                   target_dir: str = "data") -> bool:
    """
    Extract nuScenes v1.0-mini dataset.
    
    Args:
        archive_path: Path to the .tgz archive
        target_dir: Target directory for extraction
        
    Returns:
        True if extraction successful, False otherwise
    """
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    
    if not archive_path.exists():
        print(f"Error: Archive file not found at {archive_path}")
        return False
    
    print(f"Extracting {archive_path} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=target_dir)
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False


def verify_dataset_structure(data_dir: str = "data") -> Dict[str, bool]:
    """
    Verify that the nuScenes v1.0-mini dataset has the expected structure.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary with verification results for each component
    """
    data_dir = Path(data_dir)
    results = {}
    
    # Check for v1.0-mini metadata directory
    metadata_dir = data_dir / "v1.0-mini"
    results['metadata_dir'] = metadata_dir.exists()
    
    if results['metadata_dir']:
        # Check for required JSON files
        required_files = [
            'attribute.json',
            'calibrated_sensor.json',
            'category.json',
            'ego_pose.json',
            'instance.json',
            'log.json',
            'map.json',
            'sample.json',
            'sample_annotation.json',
            'sample_data.json',
            'scene.json',
            'sensor.json',
            'visibility.json'
        ]
        
        for filename in required_files:
            file_path = metadata_dir / filename
            results[f'metadata_{filename}'] = file_path.exists()
    
    # Check for samples directory
    samples_dir = data_dir / "samples"
    results['samples_dir'] = samples_dir.exists()
    
    if results['samples_dir']:
        # Check for camera directories
        camera_names = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        for cam in camera_names:
            cam_dir = samples_dir / cam
            results[f'camera_{cam}'] = cam_dir.exists()
        
        # Check for LiDAR directory
        lidar_dir = samples_dir / "LIDAR_TOP"
        results['lidar_dir'] = lidar_dir.exists()
        
        # Check for RADAR directories
        radar_names = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]
        for radar in radar_names:
            radar_dir = samples_dir / radar
            results[f'radar_{radar}'] = radar_dir.exists()
    
    # Check for sweeps directory
    sweeps_dir = data_dir / "sweeps"
    results['sweeps_dir'] = sweeps_dir.exists()
    
    # Check for maps directory
    maps_dir = data_dir / "maps"
    results['maps_dir'] = maps_dir.exists()
    
    return results


def print_verification_results(results: Dict[str, bool]) -> bool:
    """
    Print verification results in a readable format.
    
    Args:
        results: Dictionary of verification results
        
    Returns:
        True if all checks passed, False otherwise
    """
    print("\n" + "="*60)
    print("Dataset Structure Verification")
    print("="*60)
    
    all_passed = True
    for key, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {key}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("All checks passed! Dataset is ready to use.")
    else:
        print("Some checks failed. Please verify dataset extraction.")
    print("="*60 + "\n")
    
    return all_passed


def count_samples(data_dir: str = "data") -> Dict[str, int]:
    """
    Count the number of samples in the dataset.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary with sample counts
    """
    data_dir = Path(data_dir)
    counts = {}
    
    # Count scenes
    scene_file = data_dir / "v1.0-mini" / "scene.json"
    if scene_file.exists():
        with open(scene_file, 'r') as f:
            scenes = json.load(f)
            counts['scenes'] = len(scenes)
    
    # Count samples
    sample_file = data_dir / "v1.0-mini" / "sample.json"
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            samples = json.load(f)
            counts['samples'] = len(samples)
    
    # Count annotations
    annotation_file = data_dir / "v1.0-mini" / "sample_annotation.json"
    if annotation_file.exists():
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
            counts['annotations'] = len(annotations)
    
    # Count LiDAR files
    lidar_dir = data_dir / "samples" / "LIDAR_TOP"
    if lidar_dir.exists():
        counts['lidar_files'] = len(list(lidar_dir.glob("*.pcd.bin")))
    
    # Count camera images
    camera_names = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    total_images = 0
    for cam in camera_names:
        cam_dir = data_dir / "samples" / cam
        if cam_dir.exists():
            cam_count = len(list(cam_dir.glob("*.jpg")))
            counts[f'images_{cam}'] = cam_count
            total_images += cam_count
    counts['total_images'] = total_images
    
    return counts


def print_sample_counts(counts: Dict[str, int]):
    """Print sample counts in a readable format."""
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    for key, count in counts.items():
        print(f"{key}: {count}")
    print("="*60 + "\n")


def main():
    """Main function to extract and verify dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract and verify nuScenes v1.0-mini dataset"
    )
    parser.add_argument(
        '--archive',
        type=str,
        default='v1.0-mini.tgz',
        help='Path to the dataset archive (default: v1.0-mini.tgz)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='data',
        help='Target directory for extraction (default: data)'
    )
    parser.add_argument(
        '--skip-extract',
        action='store_true',
        help='Skip extraction and only verify'
    )
    parser.add_argument(
        '--force-extract',
        action='store_true',
        help='Force re-extraction even if data exists'
    )
    
    args = parser.parse_args()
    
    # Check if data already exists
    data_exists = Path(args.target).exists() and \
                  (Path(args.target) / "v1.0-mini").exists()
    
    # Extract if needed
    if not args.skip_extract:
        if args.force_extract or not data_exists:
            success = extract_dataset(args.archive, args.target)
            if not success:
                print("Extraction failed!")
                return 1
        else:
            print(f"Data already exists at {args.target}. Use --force-extract to re-extract.")
    
    # Verify structure
    results = verify_dataset_structure(args.target)
    all_passed = print_verification_results(results)
    
    # Count samples
    if all_passed:
        counts = count_samples(args.target)
        print_sample_counts(counts)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
