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


parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--iter', default='700', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 700')
parser.add_argument('--input-list', default='file_paths.txt', type=str,
                    help='input-list: path to the text file containing the input file paths. Default: ./file_paths.txt')
parser.add_argument('--output-dir', default='derived/pipeline/', type=str,
                    help='output-dir: directory to save output files. Default: derived/pipeline/')
parser.add_argument('--cuda-device', default='0', type=str,
                    help='cuda-device: which CUDA device to use. Default: 0')

opt = parser.parse_args()


os.makedirs(opt.output_dir, exist_ok=True)

# Define the log file path
log_file_path = os.path.join(opt.output_dir, 'cvd-risk-score.log')
if os.path.exists(log_file_path):
    os.remove(log_file_path)
    open(log_file_path, 'w').close()

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
        sub_part = sub_match.group(0)
        ses_match = re.search(ses_pattern, input_path)
        ses_part = ses_match.group(0)
        # output files must be stored respecting BIDS structure:
        # output_dir/sub-XXXXXX/ses-XX/anat
        output_path = os.path.join(opt.output_dir, sub_part, ses_part, 'anat')
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.basename(input_path)
        score_file_name = file_name.replace('ct.nii.gz', 'desc-cvdr.txt')
        score_file_path = os.path.join(output_path, score_file_name)
        
        heart_slices_file_name = file_name.replace('ct.nii.gz', 'desc-heartslices.txt')
        heart_slices_file_path = os.path.join(output_path, heart_slices_file_name)

        gradmap_file_name = file_name.replace('ct.nii.gz', 'desc-gradmap.png')
        gradmap_file_path = os.path.join(output_path, gradmap_file_name)

        image = Image()
        start_time = time.time()
        # image.detect_heart(input_path)
        ## added 27/03
        try:
            image.detect_heart(input_path)
        except RuntimeError as e:
            logger.error(f'Error processing {input_path}: {e}')
            # Log the error and continue to the next file
            with open(score_file_path, 'w') as output_file:
                output_file.write(f'FAILED HEART DETECTION')
            with open(heart_slices_file_path, 'w') as output_file:
                output_file.write(f'FAILED HEART DETECTION')
            continue
        ##
        if not image.heart_detected:
            logger.error(f'Heart detection failed for {input_path}. Skipping to the next image.')
            with open(score_file_path, 'w') as output_file:
                output_file.write(f'FAILED HEART DETECTION')
            with open(heart_slices_file_path, 'w') as output_file:
                output_file.write(f'FAILED HEART DETECTION')
            continue

        #image.detect_visual(output_dir=output_path)
        image_identifier = file_name.replace('_ct.nii.gz', '')
        image.detect_visual(output_dir=output_path, file_name_suffix=image_identifier)
        network_input = image.to_network_input()
        cvd_risk_score = m.aug_transform(network_input)[1]
        gradmap = m.grad_cam_visual(image.to_network_input())
        plt.savefig(gradmap_file_path)

        with open(score_file_path, 'w') as output_file:
            output_file.write(f'Estimated CVD Risk: {cvd_risk_score}\n')
        with open(heart_slices_file_path, 'w') as output_file:
            output_file.write(f'First heart slice: {image.first_slice}\n')
            output_file.write(f'Last heart slice: {image.last_slice}\n')

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f'Processed {input_path} in {elapsed_time:.2f} seconds. CVD Risk saved to {score_file_path}.')

# Close the log file
sys.stdout.close()
logger.info(f'Log file saved to: {log_file_path}')
     