# -*- coding: utf-8 -*-
# @Author  : Giulia
# @Time    : 2024/09/26

import argparse
import os
import numpy as np
from init_model import init_model
from image import Image
import time
import sys
import logging 
import re
import matplotlib.pyplot as plt
import csv


parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--iter', default='700', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 700')
parser.add_argument('--input-list', default='file_paths.txt', type=str,
                    help='input-list: path to the text file containing the input file paths. Default: ./file_paths.txt')
parser.add_argument('--output-dir', default='./derived/pipeline/', type=str,
                    help='output-dir: directory to save output files. Default: derived/pipeline/')
parser.add_argument('--cuda-device', default='0', type=str,
                    help='cuda-device: which CUDA device to use. Default: 0')
parser.add_argument('--save-maps', action='store_true',
                    help='save-maps: wheter to save png maps of the detected heart and grad maps of the risk score')
opt = parser.parse_args()


os.makedirs(opt.output_dir, exist_ok=True)

# Define the log file path
log_file_path = os.path.join(opt.output_dir, 'cvd-risk-score.log')
if os.path.exists(log_file_path):
    os.remove(log_file_path)
    open(log_file_path, 'w').close()

# Add this after your argument parsing and before the main loop
csv_file_path = os.path.join(opt.output_dir, 'cvd_results.csv')
csv_headers = ['input_path', 'cvd_risk_score', 'first_heart_slice', 'last_heart_slice', 'status', 'processing_time_seconds']

# Initialize CSV with headers
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)

def append_to_csv(csv_path, data_row):
    """Append a single row of data to the CSV file."""
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_row)

# Configure logging for both console and file separately
logger = logging.getLogger('cvd-risk-score')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log parsed arguments
logger.info(f'Parsed Arguments: {opt}')

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device

m = init_model()
m.load_model(opt.iter)

with open(opt.input_list, 'r') as file_paths:
    for line in file_paths:
        # Define regular expressions to match 'sub' and 'ses' parts
        sub_pattern = r'sub-\d+'
        ses_pattern = r'ses-[A-Za-z0-9]+'
        input_path = line.strip()  # Remove leading/trailing whitespaces and newline characters
        logger.info(f'input path: {input_path}')
        #print('input path', input_path)
        # Extract 'sub' and 'ses' parts from the file path
        sub_match = re.search(sub_pattern, input_path)
        ses_match = re.search(ses_pattern, input_path)
        if not sub_match or not ses_match:
            logger.error(f'BIDS format not found in {input_path}')
            continue
        sub_part = sub_match.group(0)
        ses_part = ses_match.group(0)
        # output files must be stored respecting BIDS structure:
        # output_dir/sub-XXXXXX/ses-XX/ct
        output_path = os.path.join(opt.output_dir, sub_part, ses_part, 'ct')
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.basename(input_path)
        score_file_name = file_name.replace('ct.nii.gz', 'desc-cvdr.txt')
        score_file_path = os.path.join(output_path, score_file_name)
        heart_slices_file_name = file_name.replace('ct.nii.gz', 'desc-heartslices.txt')
        heart_slices_file_path = os.path.join(output_path, heart_slices_file_name) 
        if opt.save_maps:
            gradmap_file_name = file_name.replace('ct.nii.gz', 'desc-gradmap.png')
            gradmap_file_path = os.path.join(output_path, gradmap_file_name)
        image = Image()
        start_time = time.time()
        # Initialize default values for CSV row
        cvd_risk_score = "N/A"
        first_heart_slice = "N/A" 
        last_heart_slice = "N/A"
        status = "fail"
        # image.detect_heart(input_path)
        ## added 27/03
        try:
            image.detect_heart(input_path)
        except RuntimeError as e:
            logger.error(f'Error processing {input_path}: {e}')
            with open(score_file_path, 'w') as output_file:
                output_file.write(f'FAILED HEART DETECTION')
            
            # Calculate elapsed time and write to CSV before continuing
            end_time = time.time()
            elapsed_time = end_time - start_time
            row = [input_path, cvd_risk_score, first_heart_slice, last_heart_slice, status, elapsed_time]
            append_to_csv(csv_file_path, row)
            continue
        ##
        if not image.heart_detected:
            logger.error(f'Heart detection failed for {input_path}. Skipping to the next image.')
            with open(score_file_path, 'w') as output_file:
                output_file.write(f'FAILED HEART DETECTION')
            # Calculate elapsed time and write to CSV before continuing
            end_time = time.time()
            elapsed_time = end_time - start_time
            row = [input_path, cvd_risk_score, first_heart_slice, last_heart_slice, status, elapsed_time]
            append_to_csv(csv_file_path, row)
            continue

        #image.detect_visual(output_dir=output_path)
        network_input = image.to_network_input()
        cvd_risk_score = m.aug_transform(network_input)[1]
        first_heart_slice = image.first_slice 
        last_heart_slice = image.last_slice 
        if opt.save_maps:
            image_identifier = file_name.replace('_ct.nii.gz', '')
            image.detect_visual(output_dir=output_path, file_name_suffix=image_identifier)
            gradmap = m.grad_cam_visual(image.to_network_input())
            plt.savefig(gradmap_file_path)
        with open(score_file_path, 'w') as output_file:
            output_file.write(f'Estimated CVD Risk: {cvd_risk_score}\n')
        with open(heart_slices_file_path, 'w') as output_file:
            output_file.write(f'First heart slice: {first_heart_slice}\n')
            output_file.write(f'Last heart slice: {last_heart_slice}\n')

        end_time = time.time()
        elapsed_time = end_time - start_time
        status = 'success'
        row = [input_path, cvd_risk_score, first_heart_slice, last_heart_slice, status, elapsed_time]
        append_to_csv(csv_file_path, row)

        logger.info(f'Processed {input_path} in {elapsed_time:.2f} seconds. CVD Risk saved to {score_file_path}.')

# Close the log file
logger.info(f'Log file saved to: {log_file_path}')
     