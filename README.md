# CUTS: A Fully Unsupervised Framework for Medical Image Segmentation
### Krishnaswamy Lab, Yale University

Rewritten and updated code base.

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name $OUR_CONDA_ENV pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda activate $OUR_CONDA_ENV
```

## Usage
```
cd ./src
conda activate $OUR_CONDA_ENV
python main.py --mode train --config ./config/baseline.yaml
```
