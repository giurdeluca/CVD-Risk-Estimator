> Check [README_original](README_original.md) for further documentation and citation!

# Overview
This application processes CT images in NIfTI format (`.nii.gz`) to:
1. **Detect and localize the heart** using automated bounding box detection
2. **Extract heart slices** and create visual detection outputs
3. **Compute CVD risk scores** using a trained deep learning model
4. **Generate grad-CAM visualizations** for model interpretability
5. **Output results in BIDS-compliant structure (and a csv with all the scores)**

## Performance
5s per image with a peak of 15GB GPU RAM on L40S.


# Installation

## Prerequisites

- Python 3.8
- PyTorch 1.8
- Computing device with GPU


## Local Installation
1. Clone repo and install the requirements
```bash
git clone <repository-url>
cd CVD-Risk-Estimator

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 11.1 support
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

2. Manually download models checkpoint
- **RetinaNet Checkpoint**: was obtained by running the original colab notebook once (see [here](https://github.com/DIAL-RPI/CVD-Risk-Estimator/blob/master/colab_run.ipynb)).
- **Tri-2DNet Checkpoint**: Download https://1drv.ms/u/s!AurT2TsSKdxQvz1aHvmxTlkDNkTz?e=8rCnJl and place in ./checkpoint/

### Docker Installation (Recommended)
```bash
# Build the Docker image
docker build -t cvd-risk-estimator .
```

# Usage

## Input Requirements

1. **CT Images**: NIfTI format (`.nii.gz`) containing LDCT chest scans
2. **File List**: Text file containing paths to input images (one per line)
3. **BIDS Structure**: Input files should follow BIDS naming convention with `sub-` and `ses-` identifiers

**Example file list (`file_paths.txt`):**
```
/path/to/sub-001/ses-01/anat/sub-001_ses-01_ct.nii.gz
/path/to/sub-002/ses-01/anat/sub-002_ses-01_ct.nii.gz
```

## Local Usage

```bash
python cvdrisk_BIDS.py \
    --input-list file_paths.txt \
    --output-dir results/ \
    --iter 700 \
    --cuda-device 0 \
    --save-maps 
```

## Docker Usage (Recommended)

```bash
docker run --gpus all \
    -v /path/to/input/data:/app/input:ro \
    -v /path/to/output:/app/output \
    -v /path/to/file_paths.txt:/app/file_paths.txt:ro \
    cvd-risk-estimator \
    --input-list /app/file_paths.txt \
    --output-dir /app/output \
    --iter 700 \
    --cuda-device 0 \
    --save-maps

```

### Testing GPU Access
```bash
# Test CUDA availability in container
docker run --gpus all -it --entrypoint="/bin/bash" cvd-risk-estimator
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-list` | `file_paths.txt` | Path to text file containing input CT image paths |
| `--output-dir` | `derived/pipeline/` | Directory to save output files |
| `--iter` | `700` | Model checkpoint iteration to load |
| `--cuda-device` | `0` | CUDA device ID to use for inference |
| `--save-maps` | False | If added, saves gradmaps and heartdetect mosaic |


## Output Structure

The tool generates outputs in BIDS-derived format:

```
output_dir/
├── sub-001/
│   └── ses-01/
│       └── anat/
│           ├── sub-001_ses-01_desc-cvdr.txt          # CVD risk score
│           ├── sub-001_ses-01_desc-heartslices.txt   # Heart slice indices
│           ├── sub-001_ses-01_desc-gradmap.png       # Grad-CAM visualization
│           └── sub-001_ses-01_desc-heartdetect.png   # Heart detection visualization
└── cvd-risk-score.log                                # Processing log
```

## Output Files Explained

- **`*_desc-cvdr.txt`**: Contains the estimated CVD risk score, a real number in \[0, 1\] indicating the estimated CVD risk.
- **`*_desc-heartslices.txt`**: First and last slice indices where heart was detected
- **`*_desc-gradmap.png`**: Grad-CAM heatmap showing model attention regions
- **`*_desc-heartdetect.png`**: 8x8 grid showing heart detection across slices
- **`cvd-risk-score.log`**: Detailed processing log with timing and status information

## Error Handling

The application includes robust error handling:
- **Heart detection failures** are logged and marked as "FAILED HEART DETECTION"
- **Processing errors** are captured and logged while continuing with remaining files
- **Failed cases** still generate output files with error status for tracking

# Modified Components
Some scripts that were in the colab_support originally (`bbox_cut.py`, `image.py`) have been edited and copied in the project directory in order to make the script run.
Also the retinanet checkpoint has been downloaded manually and copied here.
- `bbox_cut.py`, `image.py`: Adapted from original colab_support for standalone operation
- `heart_detect.py`: Integrated RetinaNet model for automated heart localization
- `cvdrisk_BIDS.py`: Main processing script with BIDS compliance and error handling

# Attribution
## Tri2D-Net for CVD Risk Estimation

[![DOI](https://zenodo.org/badge/256093026.svg)](https://zenodo.org/badge/latestdoi/256093026)

Tri2D-Net is the **first** deep learning network trained for directly estimating **overall** cardiovascular disease (CVD) risks on low dose computed tomography (LDCT). The corresponding [paper](https://www.nature.com/articles/s41467-021-23235-4) has been published on Nature Communications.

## Citation
Please cite these papers in your publications if the code helps your research:
```
@Article{chao2021deep,
  author  = {Chao, Hanqing and Shan, Hongming and Homayounieh, Fatemeh and Singh, Ramandeep and Khera, Ruhani Doda and Guo, Hengtao and Su, Timothy and Wang, Ge and Kalra, Mannudeep K. and Yan, Pingkun},
  title   = {Deep learning predicts cardiovascular disease risks from lung cancer screening low dose computed tomography},
  journal = {Nature Communications},
  year    = {2021},
  volume  = {12},
  number  = {1},
  pages   = {2963},
  url     = {https://doi.org/10.1038/s41467-021-23235-4},
}
```
Link to paper:
- [Deep Learning Predicts Cardiovascular Disease Risks from Lung Cancer Screening Low Dose Computed Tomography](https://www.nature.com/articles/s41467-021-23235-4)


## License
The source code of Tri2D-Net is licensed under a MIT-style license, as found in the [LICENSE](LICENSE) file.
This code is only freely available for non-commercial use, and may be redistributed under these conditions.
For commercial queries, please contact [Dr. Pingkun Yan](https://dial.rpi.edu/people/pingkun-yan).
