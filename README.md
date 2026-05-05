# \# MDR²: Meta-Reweighting with Dynamic Retrieval for Brain Tumor Segmentation

This repository provides a PyTorch implementation of a \*\*3D U-Net–based framework enhanced with gradient-based meta-reweighting and dynamic meta-case retrieval\*\* for robust brain tumor segmentation.



The framework is designed to address \*\*data imbalance and domain bias\*\*, particularly in \*\*adult–pediatric MRI segmentation scenarios\*\*, using the BraTS dataset.

## Project Overview

Medical image segmentation often suffers from:

\- Domain shift (adult vs. pediatric data)

\- Data imbalance

\- Noisy or heterogeneous annotations



To address these challenges, this project integrates:

\- \*\*3D U-Net segmentation backbone\*\*

\- \*\*Gradient-based meta-reweighting\*\*

\- \*\*Dynamic retrieval of representative meta-cases\*\*



* \## ✨ Key Features

\- 3D U-Net architecture for volumetric segmentation  

\- Multi-modal MRI support (T1, T2, T1ce, FLAIR)  

\- Gradient similarity–based reweighting  

\- Dynamic top-\*k\* meta-case selection  

\- Modular and reusable reweighting framework  



## Repository Structure

```text
MDR2/
├── configs.py                # argument parser and default experiment settings
├── scripts.py                # main training/inference workflow
├── train\_brats2021\_wieght.py # alternate training script (if used)
├── dataset/                  # dataset loaders and utilities
├── models/                   # model definitions and network blocks
├── utils/                    # metrics, loss, optimizer, scheduler utilities
├── reweighting/              # reusable reweighting/retrieval helper module
│   ├── \_\_init\_\_.py
│   ├── methods.py
│   └── README.md
├── data/                     # expected dataset root and split files
├── exp/                      # experiment outputs, logs, predictions
├── requirements.txt          # required Python dependencies
└── README.md                 # this file
```

## Data Layout

The project expects data in a structured directory layout. By default, `configs.py` points to:

* `data/train\_data` as the main dataset root
* `data/split/brats2021\_split.csv` for the primary adult/pediatric train/validation/test split
* `data/split/Adults\_brats2021\_split.csv` for the secondary adult-only split used in combined loading

The dataset loader reads case folders from `os.path.join(args.data\_root, args.dataset)`. With the default config, this resolves to `data/train\_data/brats2021`.

### Expected directory tree

```text
MDR2/
└── data/
    ├── train\_data/
    │   └── brats2021/
    │       ├── BraTS2021\_00000/
    │       │   ├── BraTS2021\_00000-t2f.nii.gz
    │       │   ├── BraTS2021\_00000-t1n.nii.gz
    │       │   ├── BraTS2021\_00000-t1c.nii.gz
    │       │   ├── BraTS2021\_00000-t2w.nii.gz
    │       │   ├── BraTS2021\_00000\_binaryseg.nii.gz
    │       │   └── BraTS2021\_00000-seg.nii.gz
    │       ├── BraTS2021\_00001/
    │       │   └── ...
    │       └── ...
    └── split/
        ├── brats2021\_split.csv
        └── Adults\_brats2021\_split.csv
```

### Split files

The split CSV files are simple two-column files with a header row:

```csv
name,split
BraTS2021\_00000,train
BraTS2021\_00001,val
BraTS2021\_00002,test
```

Valid `split` values are:

* `train`
* `val`
* `test`
* `meta\_train` (optional)

The `load\_cases\_split` utility reads the CSV and returns four lists:

* `train\_cases`
* `val\_cases`
* `test\_cases`
* `meta\_cases`

Each value in `name` should match a case folder under `data/train\_data/brats2021/`.

### Supported case file naming

The dataset loader supports two case formats in this repository:

* `BraTS2021Dataset` (default adult/pediatric dataset format):

  * `<case\_name>-t2f.nii.gz`
  * `<case\_name>-t1n.nii.gz`
  * `<case\_name>-t1c.nii.gz`
  * `<case\_name>-t2w.nii.gz`
  * `<case\_name>\_binaryseg.nii.gz`
  * `<case\_name>-seg.nii.gz`
* `Adults\_BraTS2021Dataset` (alternate format):

  * `<case\_name>\_flair.nii.gz`
  * `<case\_name>\_t1.nii.gz`
  * `<case\_name>\_t1ce.nii.gz`
  * `<case\_name>\_t2.nii.gz`
  * `<case\_name>\_binaryseg.nii.gz`
  * `<case\_name>\_seg.nii.gz`

The split CSV files should list case names matching the folder names inside `data/train\_data/brats2021/`.

## Installation

### 1\. Clone the repository

```bash
git clone https://github.com/yourusername/3DUNet-BraTS-PyTorch.git
cd 3DUNet-BraTS-PyTorch-master/MDR2
```

### 2\. Create a virtual environment

Use your preferred environment manager. Example with `venv`:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # Linux/macOS
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Install PyTorch

Install the correct PyTorch version for your CUDA support. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you use CPU only, follow the official PyTorch installation instructions for your platform.

## Usage

### Training

Run the main training workflow:

```bash
python scripts.py
```

The script uses the default settings from `configs.py`. To customize training, pass additional arguments, for example:

```bash
python scripts.py --epochs 100 --batch\_size 8 --lr 1e-4 --comment my\_experiment
```

### Inference

The same `scripts.py` file includes a validation/test inference pipeline in its training loop and final testing phase.

### Experiment outputs

During execution, experiment outputs are saved under `exp/` and include:

* model checkpoints
* TensorBoard logs
* prediction results
* validation metrics

## Reweighting and Retrieval Module

The folder `reweighting/` contains reusable gradient-based reweighting and retrieval utilities for external use:

* `methods.py`: helper functions for gradient computation, sample gradient extraction, reference gradient averaging, top-k sample selection, and meta-weight computation.
* `README.md`: quick usage guide for copy/paste integration.

Example import:

```python
from reweighting import (
    compute\_reference\_gradient,
    select\_topk\_meta\_cases,
    compute\_meta\_weights,
)
```

## Configuration Options

The project uses `configs.py` for default arguments including:

* dataset root and split file paths
* batch size, epochs, learning rate
* optimizer and scheduler settings
* model architecture and input channel count
* inference patch size, overlap, and sliding window mode

## Notes

* Make sure your data paths match the defaults in `configs.py`, or override them with command-line arguments.
* If you use GPU, verify CUDA compatibility with the installed PyTorch version.
* The project currently targets BraTS-style MRI segmentation with 3 output classes.


## Acknowledgments

This work was supported by the Korea Institute of Science and Technology (KIST) Institutional Program under Grant No. 26E0212, and 26E0214. This work was also supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00561616).

