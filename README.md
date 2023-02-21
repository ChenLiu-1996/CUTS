# CUTS: A Fully Unsupervised Framework for Medical Image Segmentation

### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/UnsupervisedMedicalSeg.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/UnsupervisedMedicalSeg/)


<img src = "assets/github_img1.png" width=720>
<img src = "assets/github_img2.png" width=700>


**This repository contains the official PyTorch implementation of the following paper:**

> **CUTS: A Fully Unsupervised Framework for Medical Image Segmentation**<br>
> Chen Liu, Matthew Amodio, Liangbo L. Shen, Feng Gao, Arman Avesta, Sanjay Aneja, Jay Wang, Lucian V. Del Priore, Smita Krishnaswamy <br>
>
>
> *Chen and Matthew are co-first authors and Sanjay, Jay, Lucian and Smita are co-advisory authors.*
>
> Please direct correspondence to: smita.krishnaswamy@yale.edu or lucian.delpriore@yale.edu.
>
> https://arxiv.org/abs/2209.11359


## Repository Hierarchy
```
UnsupervisedMedicalSeg (CUTS)
    ├── checkpoints: model weights are saved here.
    ├── config: configuration yaml files.
    ├── data: folders containing data used.
    ├── logs: training log files.
    ├── results: generated results (images, labels, segmentations, figures, etc.).
    └── src
        ├── data_utils
        ├── datasets: defines how to access and process the data in `CUTS/data/`.
        ├── model
        ├── scripts_analysis: scripts for analysis and plotting.
        |   ├── `generate_baselines.py`
        |   ├── `generate_diffusion.py`
        |   ├── `generate_kmeans.py`
        |   ├── `helper_generate_kmeans.py`
        |   ├── `helper_run_phate.py`
        |   ├── `plot_paper_figure_medical.py`
        |   ├── `plot_paper_figure_natural.py`
        |   └── `run_metrics.py`
        ├── utils
        ├── `main_supervised.py`: supervised training of UNet/nnUNet for comparison.
        └── `main.py`: unsupervised training of the CUTS encoder.
```


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
python -m pip install opencv-python
python -m pip install sewar
python -m pip install monai
python -m pip install nibabel
```
Installation usually takes between 20 minutes and 1 hour on a normal desktop computer.

## Usage
<details>
  <summary>Unzip data</summary>

```
cd ./data/
unzip retina.zip
```
</details>

<details>
  <summary>Activate environment</summary>

```
conda activate $OUR_CONDA_ENV
```
</details>

<details>
  <summary><b>Stage 1.</b> Training the convolutional encoder</summary>

#### To train a model.
```
## Under $CUTS_ROOT/src
python main.py --mode train --config ../config/$CONFIG_FILE.yaml
```
#### To test a model (automatically done during `train` mode).
```
## Under $CUTS_ROOT/src
python main.py --mode test --config ../config/$CONFIG_FILE.yaml
```
</details>

<details>
  <summary>(Optional) [Comparison] Training a supervised model</summary>

```
## Under $CUTS_ROOT/src/
python main_supervised.py --mode train --config ../$CONFIG_FILE.yaml
```
</details>


<details>
  <summary><b>Stage 2.</b> Results Generation</summary>

#### To generate and save the segmentation using spectral k-means.
```
## Under $CUTS_ROOT/src/scripts_analysis
python generate_kmeans.py --config ../../config/$CONFIG_FILE.yaml
```
#### To generate and save the segmentation using diffusion condensation.
```
## Under $CUTS_ROOT/src/scripts_analysis
python generate_diffusion.py --config ../../config/$CONFIG_FILE.yaml
```
#### To generate and save the segmentation using baseline methods.
```
## Under $CUTS_ROOT/src/scripts_analysis
python generate_baselines.py
```
</details>

<details>
  <summary>Results Plotting</summary>

#### To reproduce the figures in the paper.
Note: This is a newer version for plotting, and it already entails the following versions (spectral k-means, diffusion condensation). You don't need to worry about them if you use this plotting script.

Without the `--comparison` flag, the CUTS-only results will be plotted.
With the ` --comparison` flag, the side-by-side comparison against other methods will be plotted.
```
## Under $CUTS_ROOT/src/scripts_analysis

## For natural images (berkeley)
python plot_paper_figure_natural.py --config ../../config/$CONFIG_FILE.yaml --image-idx $IMAGE_IDX
python plot_paper_figure_natural.py --config ../../config/$CONFIG_FILE.yaml --image-idx $IMAGE_IDX --comparison

## For medical images (retina, brain)
python plot_paper_figure_medical.py --config ../../config/$CONFIG_FILE.yaml --image-idx $IMAGE_IDX
python plot_paper_figure_medical.py --config ../../config/$CONFIG_FILE.yaml --image-idx $IMAGE_IDX --comparison
```
</details>

<details>
  <summary>Results Analysis</summary>

#### To compute the quantitative metrics.
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python run_metrics.py --config ../../config/$CONFIG_FILE.yaml
```
</details>


## DEBUG Notes
<details>
  <summary>Regarding occasional "deadlock" when generating results (especially for generating k-means and plotting figures).</summary>

On our YCRC server, sometimes we need to run
```
export MKL_THREADING_LAYER=GNU
```
before running some of the code code to minimize the risk of dead lock. For details, see https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md.

**UPDATE Dec 26, 2022**: I finally wrote a workaround to avoid running the script over and over again from the first incomplete file whenever a deadlock is hit (which is a total waste of human efforts)! The method is simple: in `generate_kmeans.py` we now outsource the kmeans computation and numpy saving to a helper file `helper_generate_kmeans.py`, and we kill and restart the helper whenever a deadlock causes the process to timeout. **However**, on our YCRC server, you may **still** need to run the command `export MKL_THREADING_LAYER=GNU` to minimize risk of dead lock.

</details>
