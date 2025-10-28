#!/usr/bin/env python3
"""
Script to read and summarize H5 data files in the splits directory.
Provides information about data shapes, keys, and contents of each H5 file.
"""

import h5py
import numpy as np
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Tuple


def analyze_h5_group(group, path="", max_depth=3, current_depth=0):
    """
    Recursively analyze an H5 group and return information about its contents.
    
    Args:
        group: H5 group or file object
        path: Current path in the H5 hierarchy
        max_depth: Maximum depth to traverse
        current_depth: Current traversal depth
    
    Returns:
        Dictionary with analysis results
    """
    info = {}
    
    if current_depth >= max_depth:
        return {"note": "Max depth reached"}
    
    for key in group.keys():
        item_path = f"{path}/{key}" if path else key
        item = group[key]
        
        if isinstance(item, h5py.Dataset):
            # Analyze dataset
            info[key] = {
                "type": "dataset",
                "shape": item.shape,
                "dtype": str(item.dtype),
                "size": item.size,
                "path": item_path
            }
            
            # Add sample data for small datasets or first few elements for large ones
            if item.size <= 100:
                try:
                    info[key]["sample_data"] = item[...].tolist() if item.size <= 10 else "Large dataset - use item[...] to view"
                except:
                    info[key]["sample_data"] = "Could not read sample data"
            else:
                try:
                    # Show shape and first few elements for large arrays
                    if len(item.shape) == 1:
                        info[key]["first_elements"] = item[:min(5, item.shape[0])].tolist()
                    elif len(item.shape) == 2:
                        info[key]["first_elements"] = item[:min(3, item.shape[0]), :min(5, item.shape[1])].tolist()
                    else:
                        info[key]["first_elements"] = "Multi-dimensional array - shape: " + str(item.shape)
                except:
                    info[key]["first_elements"] = "Could not read sample data"
            
            # Add attributes if any
            if item.attrs:
                info[key]["attributes"] = dict(item.attrs)
                
        elif isinstance(item, h5py.Group):
            # Recursively analyze group
            info[key] = {
                "type": "group",
                "path": item_path,
                "contents": analyze_h5_group(item, item_path, max_depth, current_depth + 1)
            }
            
            # Add attributes if any
            if item.attrs:
                info[key]["attributes"] = dict(item.attrs)
    
    return info


def summarize_h5_file(filepath: Path) -> Dict[str, Any]:
    """
    Analyze a single H5 file and return summary information.
    
    Args:
        filepath: Path to the H5 file
    
    Returns:
        Dictionary with file analysis
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Get basic file info
            summary = {
                "filepath": str(filepath),
                "file_size_mb": filepath.stat().st_size / (1024 * 1024),
                "keys": list(f.keys()),
                "attributes": dict(f.attrs) if f.attrs else {},
                "structure": analyze_h5_group(f)
            }
            
        return summary
        
    except Exception as e:
        return {
            "filepath": str(filepath),
            "error": str(e),
            "file_size_mb": filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
        }


def analyze_splits_directory(splits_dir: Path) -> Dict[str, Any]:
    """
    Analyze all H5 files in the splits directory structure.
    
    Args:
        splits_dir: Path to the splits directory
    
    Returns:
        Comprehensive analysis of all H5 files
    """
    analysis = {
        "directory": str(splits_dir),
        "subdirectories": {},
        "summary": {
            "total_files": 0,
            "total_size_mb": 0,
            "split_types": set(),
            "data_splits": []
        }
    }
    
    # Find all subdirectories with H5 data
    for subdir in sorted(splits_dir.iterdir()):
        print(f"Checking directory: {subdir.name}")
        if subdir.is_dir() and "splits" in subdir.name:
            print(f"  Found splits directory: {subdir.name}")
            subdir_analysis = {
                "path": str(subdir),
                "files": {}
            }
            
            # Analyze each H5 file in the subdirectory
            h5_files = list(subdir.glob("*.h5"))
            print(f"  Found {len(h5_files)} H5 files: {[f.name for f in h5_files]}")
            
            for h5_file in sorted(h5_files):
                print(f"    Analyzing: {h5_file.name}")
                file_analysis = summarize_h5_file(h5_file)
                subdir_analysis["files"][h5_file.name] = file_analysis
                
                # Update summary statistics
                analysis["summary"]["total_files"] += 1
                analysis["summary"]["total_size_mb"] += file_analysis.get("file_size_mb", 0)
                analysis["summary"]["split_types"].add(h5_file.stem)  # train, val, test
            
            analysis["subdirectories"][subdir.name] = subdir_analysis
            analysis["summary"]["data_splits"].append(subdir.name)
    
    # Convert set to list for JSON serialization
    analysis["summary"]["split_types"] = sorted(list(analysis["summary"]["split_types"]))
    
    return analysis


def print_summary(analysis: Dict[str, Any]):
    """
    Print a human-readable summary of the H5 analysis.
    
    Args:
        analysis: Analysis results from analyze_splits_directory
    """
    print("=" * 80)
    print("H5 DATA ANALYSIS SUMMARY")
    print("=" * 80)
    
    summary = analysis["summary"]
    print(f"Directory: {analysis['directory']}")
    print(f"Total H5 files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")
    print(f"Data splits found: {', '.join(summary['data_splits'])}")
    print(f"Split types: {', '.join(summary['split_types'])}")
    print()
    
    # Detailed analysis for each subdirectory
    for subdir_name, subdir_data in analysis["subdirectories"].items():
        print(f"\n{'-' * 60}")
        print(f"SPLIT: {subdir_name}")
        print(f"{'-' * 60}")
        
        for filename, file_data in subdir_data["files"].items():
            print(f"\nFile: {filename}")
            print(f"  Size: {file_data.get('file_size_mb', 0):.2f} MB")
            
            if "error" in file_data:
                print(f"  ERROR: {file_data['error']}")
                continue
                
            print(f"  Top-level keys: {file_data.get('keys', [])}")
            
            # Print structure information
            structure = file_data.get("structure", {})
            for key, info in structure.items():
                if info.get("type") == "dataset":
                    print(f"    {key}:")
                    print(f"      Shape: {info['shape']}")
                    print(f"      Data type: {info['dtype']}")
                    print(f"      Size: {info['size']} elements")
                    
                    if "first_elements" in info:
                        print(f"      Sample: {info['first_elements']}")
                    elif "sample_data" in info:
                        print(f"      Sample: {info['sample_data']}")
                        
                elif info.get("type") == "group":
                    print(f"    {key}: [GROUP]")
                    # Print group contents summary
                    contents = info.get("contents", {})
                    dataset_count = sum(1 for v in contents.values() if v.get("type") == "dataset")
                    group_count = sum(1 for v in contents.values() if v.get("type") == "group")
                    print(f"      Contains: {dataset_count} datasets, {group_count} groups")


def compare_splits(analysis: Dict[str, Any]):
    """
    Compare data shapes and structures across different splits.
    
    Args:
        analysis: Analysis results from analyze_splits_directory
    """
    print("\n" + "=" * 80)
    print("CROSS-SPLIT COMPARISON")
    print("=" * 80)
    
    # Collect data shapes for each split type (train, val, test)
    split_comparison = {}
    
    for subdir_name, subdir_data in analysis["subdirectories"].items():
        for filename, file_data in subdir_data["files"].items():
            split_type = filename.replace(".h5", "")  # train, val, test
            
            if split_type not in split_comparison:
                split_comparison[split_type] = {}
            
            if "structure" in file_data:
                for key, info in file_data["structure"].items():
                    if info.get("type") == "dataset":
                        if key not in split_comparison[split_type]:
                            split_comparison[split_type][key] = []
                        
                        split_comparison[split_type][key].append({
                            "split": subdir_name,
                            "shape": info["shape"],
                            "dtype": info["dtype"],
                            "size": info["size"]
                        })
    
    # Print comparison
    for split_type in sorted(split_comparison.keys()):
        print(f"\n{split_type.upper()} SPLIT COMPARISON:")
        print("-" * 40)
        
        for dataset_name in sorted(split_comparison[split_type].keys()):
            print(f"\n  Dataset: {dataset_name}")
            for entry in split_comparison[split_type][dataset_name]:
                print(f"    {entry['split']}: shape={entry['shape']}, size={entry['size']}, dtype={entry['dtype']}")


def create_text_report(splits_dir: Path) -> str:
    """
    Create a simple text report of H5 file contents.
    
    Args:
        splits_dir: Path to the splits directory
    
    Returns:
        String containing the text report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("H5 DATA STRUCTURE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    found_any = False
    
    # First check if there are H5 files directly in the splits directory
    h5_files_direct = list(splits_dir.glob("*.h5"))
    if h5_files_direct:
        found_any = True
        report_lines.append(f"DIRECTORY: {splits_dir.name}/")
        report_lines.append("-" * 60)
        
        # Analyze each H5 file
        for h5_file in sorted(h5_files_direct):
            try:
                with h5py.File(h5_file, 'r') as f:
                    file_size_mb = h5_file.stat().st_size / (1024 * 1024)
                    num_reactions = len(f.keys())
                    
                    report_lines.append(f"\n{h5_file.name}:")
                    report_lines.append(f"  Reactions: {num_reactions}")
                    report_lines.append(f"  File size: {file_size_mb:.1f} MB")
                    
                    # Get structure from first reaction
                    if num_reactions > 0:
                        first_rxn = list(f.keys())[0]
                        datasets = list(f[first_rxn].keys())
                        report_lines.append(f"  Datasets per reaction: {datasets}")
                        
                        # Show shapes and dtypes
                        for ds_name in sorted(datasets):
                            dataset = f[first_rxn][ds_name]
                            shape = dataset.shape
                            dtype = str(dataset.dtype)
                            report_lines.append(f"    {ds_name}: {shape} ({dtype})")
                    
            except Exception as e:
                report_lines.append(f"\n{h5_file.name}: ERROR - {str(e)}")
        
        report_lines.append("")
    
    # Also check subdirectories with H5 data
    for subdir in sorted(splits_dir.iterdir()):
        if subdir.is_dir():
            h5_files_subdir = list(subdir.glob("*.h5"))
            if h5_files_subdir:
                found_any = True
                report_lines.append(f"SUBDIRECTORY: {subdir.name}/")
                report_lines.append("-" * 60)
                
                # Analyze each H5 file in the subdirectory
                for h5_file in sorted(h5_files_subdir):
                    try:
                        with h5py.File(h5_file, 'r') as f:
                            file_size_mb = h5_file.stat().st_size / (1024 * 1024)
                            num_reactions = len(f.keys())
                            
                            report_lines.append(f"\n{h5_file.name}:")
                            report_lines.append(f"  Reactions: {num_reactions}")
                            report_lines.append(f"  File size: {file_size_mb:.1f} MB")
                            
                            # Get structure from first reaction
                            if num_reactions > 0:
                                first_rxn = list(f.keys())[0]
                                datasets = list(f[first_rxn].keys())
                                report_lines.append(f"  Datasets per reaction: {datasets}")
                                
                                # Show shapes and dtypes
                                for ds_name in sorted(datasets):
                                    dataset = f[first_rxn][ds_name]
                                    shape = dataset.shape
                                    dtype = str(dataset.dtype)
                                    report_lines.append(f"    {ds_name}: {shape} ({dtype})")
                            
                    except Exception as e:
                        report_lines.append(f"\n{h5_file.name}: ERROR - {str(e)}")
                
                report_lines.append("")
    
    if not found_any:
        report_lines.append("No H5 files found.")
        report_lines.append("")
        report_lines.append("Available items in directory:")
        for item in sorted(splits_dir.iterdir()):
            if item.is_dir():
                h5_count = len(list(item.glob("*.h5")))
                report_lines.append(f"  [DIR]  {item.name}/ ({h5_count} H5 files)")
            else:
                report_lines.append(f"  [FILE] {item.name}")
    
    return "\n".join(report_lines)


def main():
    """Main function to run the H5 analysis."""
    # Set up paths
    script_dir = Path(__file__).parent
    splits_dir = script_dir.parent / "splits"
    
    print(f"Script directory: {script_dir}")
    print(f"Looking for splits directory at: {splits_dir}")
    
    if not splits_dir.exists():
        print(f"Error: Splits directory not found at {splits_dir}")
        print("Available directories:")
        parent_dir = script_dir.parent
        for item in parent_dir.iterdir():
            if item.is_dir():
                print(f"  {item.name}")
        return
    
    print("Analyzing H5 files in splits directory...\n")
    
    # Run full analysis first
    analysis = analyze_splits_directory(splits_dir)
    
    # Print detailed summary
    print_summary(analysis)
    
    # Compare across splits
    compare_splits(analysis)
    
    # Create and display text report
    report = create_text_report(splits_dir)
    print("\n" + "=" * 80)
    print("TEXT REPORT")
    print("=" * 80)
    print(report)
    
    # Save text report to file
    output_file = splits_dir / "h5_structure_report.txt"
    try:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving report: {e}")
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
