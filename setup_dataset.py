#!/usr/bin/env python3
"""
Dataset Setup Script for TSP Solver

Downloads the TSPLIB95 dataset and filters for TSP instances
with Euclidean 2D edge weights, extracting them to a local dataset directory.
"""

import tarfile
import gzip
import requests
import re
import io
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SEP = "\n" + "=" * 40 + "\n"

def main():
    """Main function to set up the TSP dataset."""
    # Configuration
    DATASET_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
    DATASET_DIR = Path("dataset")
    
    logger.info("TSP Dataset Setup Script")
    print(SEP)
    
    # Create dataset directory
    DATASET_DIR.mkdir(exist_ok=True)
    
    try:
        logger.info(f"Downloading {DATASET_URL}...")
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        logger.info("Download completed")
        
        # Load tar.gz into memory
        tar_bytes = io.BytesIO(response.content)
        
        logger.info("Processing files...")
        extracted_files = []
        
        with tarfile.open(fileobj=tar_bytes, mode='r:gz') as tar:
            tsp_gz_files = [member for member in tar.getmembers() if member.name.endswith('.tsp.gz')]
            tour_gz_files = [member for member in tar.getmembers() if member.name.endswith('.opt.tour.gz')]
            
            # First pass: Collect valid TSP files
            valid_tsp_names = set()
            for gz_member in tsp_gz_files:
                try:
                    gz_content = tar.extractfile(gz_member).read()
                    decompressed_content = gzip.decompress(gz_content).decode('utf-8')
                    
                    has_type_tsp = bool(re.search(r'TYPE:\s*TSP', decompressed_content))
                    has_euc_2d = bool(re.search(r'EDGE_WEIGHT_TYPE:\s*EUC_2D', decompressed_content))
                    
                    if has_type_tsp and has_euc_2d:
                        tsp_filename = gz_member.name.split('/')[-1].replace('.gz', '')
                        output_file = DATASET_DIR / tsp_filename
                        
                        with open(output_file, 'w') as f:
                            f.write(decompressed_content)
                        
                        extracted_files.append(output_file)
                        base_name = tsp_filename.replace('.tsp', '')
                        valid_tsp_names.add(base_name)
                        
                except Exception as e:
                    logger.error(f"Failed to process {gz_member.name}: {e}")
            
            # Second pass: Collect tour files
            for gz_member in tour_gz_files:
                try:
                    tour_filename = gz_member.name.split('/')[-1].replace('.gz', '')
                    base_name = tour_filename.replace('.opt.tour', '')
                    
                    if base_name in valid_tsp_names:
                        gz_content = tar.extractfile(gz_member).read()
                        
                        decompressed_content = gzip.decompress(gz_content).decode('utf-8')
                        
                        output_file = DATASET_DIR / tour_filename
                        
                        with open(output_file, 'w') as f:
                            f.write(decompressed_content)
                        
                        extracted_files.append(output_file)

                except Exception as e:
                    logger.error(f"Failed to process {gz_member.name}: {e}")
    
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise e
    
    print(SEP)
    logger.info(f"Extracted files to: {DATASET_DIR}")
    
    tsp_files = [f for f in extracted_files if f.name.endswith('.tsp')]
    tour_files = [f for f in extracted_files if f.name.endswith('.opt.tour')]
    
    logger.info(f"TSP files ({len(tsp_files)}):")
    for file in sorted(tsp_files):
        logger.info(f"  - {file.name}")
    
    logger.info(f"Tour files ({len(tour_files)}):")
    for file in sorted(tour_files):
        logger.info(f"  - {file.name}")

        
if __name__ == "__main__":
    main()