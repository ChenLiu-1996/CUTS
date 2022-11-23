# CUTS: A Fully Unsupervised Framework for Medical Image Segmentation
### Krishnaswamy Lab, Yale University

Rewritten and updated code base.

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name $OUR_CONDA_ENV pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate $OUR_CONDA_ENV
conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda
python -m pip install -U phate
python -m pip install git+https://github.com/KrishnaswamyLab/CATCH

# See https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md
conda install numpy scipy -c conda-forge
```

## Usage
```
cd $CUTS_ROOT/src
conda activate $OUR_CONDA_ENV
python main.py --mode train --config ../config/retina.yaml
```
