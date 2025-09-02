# -*- coding: utf-8 -*-
# @Author  : Giulia
# @Time    : 2023/10/10
import argparse
import os
import numpy as np
from init_model import init_model
from image import Image

parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--iter', default='700', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 700')
parser.add_argument('--path', default='./demos/Positive_CAC_1.nii.gz', type=str,
                    help='path: path of the input NIfTI file. Default: ./demos/Positive_CAC_1.nii.gz')
parser.add_argument('--output-dir', default=None, type=str, help='output-dir: directory to save output files. Default: None')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

m = init_model()
m.load_model(opt.iter)
#m = torch.load('model.pth')
image = Image()
image.detect_heart(opt.path)
image.detect_visual(output_dir=opt.output_dir)
network_input = image.to_network_input()
cvd_risk_score = m.aug_transform(network_input)[1]

print('Estimated CVD Risk:', cvd_risk_score)