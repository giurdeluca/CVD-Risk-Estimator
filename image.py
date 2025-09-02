# -*- coding: utf-8 -*-
# @Author  : Giulia
# @Time    : 2023/03/25
import io
import os
import os.path as osp
import SimpleITK as sitk
sitk.ProcessObject.SetGlobalDefaultThreader("platform")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import gaussian_filter
from bbox_cut import crop_w_bbox
from heart_detect import detector
from utils import norm, CT_resize


class Image:
    CT_AXIAL_SIZE = 512

    def __init__(self):
        self.org_ct_img = None
        self.bbox = None
        self.bbox_selected = None
        self.visual_bbox = None
        self.detected_ct_img = None
        self.detected_npy = None
        self.heart_detected = None # added by Giulia
        self.first_slice = None  # added by Giulia, 25/03
        self.last_slice = None   # added by Giulia, 25/03

    def detect_heart(self, file_path):
        self.org_ct_img = sitk.ReadImage(file_path)
        print('detect_heart file path', file_path)

        # Resize org ct
        old_size = np.asarray(self.org_ct_img.GetSize()).astype('float')
        if min(old_size[0], old_size[1]) < 480 or max(old_size[0], old_size[1]) > 550:
            print('Resizing the image...')
            new_size = np.asarray([
                Image.CT_AXIAL_SIZE, Image.CT_AXIAL_SIZE, old_size[-1]]
            ).astype('float')
            old_space = np.asarray(self.org_ct_img.GetSpacing()).astype('float')
            new_space = old_space * old_size / new_size
            self.org_ct_img = CT_resize(
                self.org_ct_img,
                new_size=new_size.astype('int').tolist(),
                new_space=new_space.tolist())
        self.org_npy = sitk.GetArrayFromImage(self.org_ct_img)
        self.org_npy = norm(self.org_npy, -500, 500)
        
        # detect heart
        self.bbox, self.bbox_selected, self.visual_bbox = detector(self.org_npy)

        # modified by Giulia
        if self.bbox is None or self.bbox_selected is None:  # Heart detection failed
            print('Fail to detect heart in the image. '
              'Please manually crop the heart region.')
            self.heart_detected = False
        else:
            self.heart_detected = True
            #print('visual bbox', self.visual_bbox)
            self.detected_ct_img = crop_w_bbox(
            self.org_ct_img, self.bbox, self.bbox_selected)
            #print('detected ct img', self.detected_ct_img)
            ##
        # Check if crop_w_bbox failed
            if self.detected_ct_img is None:
        #        print('crop_w_bbox failed to return a valid SimpleITK image.')
                self.detected_npy = None
                self.heart_detected = False
            else:
                self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
                self.detected_npy = norm(self.detected_npy, -300, 500)
        #
                ## added 25/03 by Giulia, to extract indices of sliced where heart is detected
                # nonzero is a tuple of two arrays so i select just the first one
                nonzero = np.nonzero(self.bbox)[0]
                self.first_slice = np.min(nonzero)
                self.last_slice = np.max(nonzero)
                #print("Heart detect, first slice:", self.first_slice)
                #print("Heart detect, last slice:", self.last_slice)
                ##

        ## previous code    
        #self.detected_ct_img = crop_w_bbox(
        #    self.org_ct_img, self.bbox, self.bbox_selected)
        

        #if self.detected_ct_img is None:
        #    print('Fail to detect heart in the image. '
        #          'Please manually crop the heart region.')
        #return

        #self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
        #self.detected_npy = norm(self.detected_npy, -300, 500)
        ## 

    # modified by Giulia           
    # def detect_visual(self, output_dir=None):
    def detect_visual(self, output_dir=None, file_name_suffix=""):
            total_img_num = len(self.visual_bbox)
            fig = plt.figure(figsize=(15, 15))
            grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
            for i in range(64):
                grid[i].imshow(self.visual_bbox[i * int(total_img_num / 64)])
            
            # Determine the output file path within the specified output directory
            # output_file = "heart_detect.png"
            output_file = f"{file_name_suffix}_desc-heartdetect.png"
            if output_dir:
                output_file = os.path.join(output_dir, output_file)
            
            # Save the image to the specified output file
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()

    def to_network_input(self):
        data = self.detected_npy
        mask = np.clip(
            (data > 0.1375).astype('float') * (data < 0.3375).astype('float')
            + (data > 0.5375).astype('float'), 0, 1)
        mask = gaussian_filter(mask, sigma=3)
        network_input = np.stack([data, data * mask]).astype('float32')
        return network_input
