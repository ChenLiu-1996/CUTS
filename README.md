<h1 align="center">
[MICCAI 2024] CUTS
</h1>

<p align="center">
<strong>A Deep Learning and Topological Framework for Multigranular Unsupervised Medical Image Segmentation</strong>
</p>

<div align="center">

[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/UnsupervisedMedicalSeg.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/UnsupervisedMedicalSeg/)
[![ArXiv](https://img.shields.io/badge/ArXiv-CUTS-firebrick)](https://arxiv.org/abs/2209.11359)
[![MICCAI 2024](https://img.shields.io/badge/MICCAI_2024-aeeafc)](https://link.springer.com/chapter/10.1007/978-3-031-72111-3_15)
[![Poster](https://img.shields.io/badge/Poster-0f4d92)](https://www.chenliu1996.com/publication/2024_cuts/CUTS_MICCAI2024_poster.pdf)

</div>

<p align="center">
Krishnaswamy Lab, Yale University
</p>


This is the authors' PyTorch implementation of [**CUTS**](https://arxiv.org/abs/2209.11359), MICCAI 2024.

The official version is maintained in the [Lab GitHub repo](https://github.com/KrishnaswamyLab/CUTS).

## A Glimpse into the Methods
<img src = "assets/architecture.png" width=800>


## Citation
```
@inproceedings{Liu_CUTS_MICCAI2024,
    title = { { CUTS: A Deep Learning and Topological Framework for Multigranular Unsupervised Medical Image Segmentation } },
    author = { Liu, Chen and Amodio, Matthew and Shen, Liangbo L. and Gao, Feng and Avesta, Arman and Aneja, Sanjay and Wang, Jay C. and Del Priore, Lucian V. and Krishnaswamy, Smita},
    booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
    publisher = {Springer Nature Switzerland},
    volume = {LNCS 15008},
    page = {155–165},
    year = {2024},
    month = {October},
}
```

## Repository Hierarchy
```
UnsupervisedMedicalSeg (CUTS)
    ├── (*) comparison: other SOTA unsupervised methods for comparison.
    |
    ├── checkpoints: model weights are saved here.
    ├── config: configuration yaml files.
    ├── data: folders containing data files.
    ├── logs: training log files.
    ├── results: generated results (images, labels, segmentations, figures, etc.).
    |
    └── src
        ├── (*) scripts_analysis: scripts for analysis and plotting.
        |   ├── `generate_baselines.py`
        |   ├── `generate_kmeans.py`
        |   ├── `generate_diffusion.py`
        |   ├── `plot_paper_figure_main.py`
        |   └── `run_metrics.py`
        |
        ├── (*) `main.py`: unsupervised training of the CUTS encoder.
        ├── (*) `main_supervised.py`: supervised training of UNet/nnUNet for comparison.
        |
        ├── datasets: defines how to access and process the data in `CUTS/data/`.
        ├── data_utils
        ├── model
        └── utils

Relatively core files or folders are marked with (*).
```

## Data Provided
The `berkeley_natural_images` and `retina` datasets are provided in `zip` format. The `brain_ventricles` dataset exceeds the GitHub size limits, and can be made available upon reasonable request.

## To reproduce the results in the paper.
The following commands are using `retina_seed2` as an example (retina dataset, random seed set to 2022).

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
conda activate cuts
```
</details>

<details>
  <summary><b>Stage 1.</b> Training the convolutional encoder</summary>

#### To train a model.
```
## Under `src`
python main.py --mode train --config ../config/retina_seed2.yaml
```
#### To test a model (automatically done during `train` mode).
```
## Under `src`
python main.py --mode test --config ../config/retina_seed2.yaml
```
</details>

<details>
  <summary>(Optional) [Comparison] Training a supervised model</summary>

```
## Under `src/`
python main_supervised.py --mode train --config ../retina_seed2.yaml
```
</details>

<details>
  <summary>(Optional) [Comparison] Training other models</summary>

#### To train STEGO.
```
## Under `comparison/STEGO/CUTS_scripts/`
python step01_prepare_data.py --config ../../../config/retina_seed2.yaml
python step02_precompute_knns.py --train-config ./train_config/train_config_retina_seed2.yaml
python step03_train_segmentation.py --train-config ./train_config/train_config_retina_seed2.yaml
python step04_produce_results.py --config ../../../config/retina_seed2.yaml --eval-config ./eval_config/eval_config_retina_seed2.yaml
```

#### To train Differentiable Feature Clustering (DFC).
```
## Under `comparison/DFC/CUTS_scripts/`
python step01_produce_results.py --config ../../../config/retina_seed2.yaml
```

#### To use Segment Anything Model (SAM).
```
## Under `comparison/SAM/`
mkdir SAM_checkpoint && cd SAM_checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Under `comparison/SAM/CUTS_scripts/`
python step01_produce_results.py --config ../../../config/retina_seed2.yaml
```

#### To use MedSAM.
```
## Under `comparison/MedSAM/`
mkdir MedSAM_checkpoint && cd MedSAM_checkpoint
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view

## Under `comparison/SAM_Med2D/CUTS_scripts/`
python step01_produce_results.py --config ../../../config/retina_seed2.yaml
```

#### To use SAM-Med2D.
```
## Under `comparison/SAM_Med2D/`
mkdir SAM_Med2D_checkpoint && cd SAM_Med2D_checkpoint
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view

## Under `comparison/SAM_Med2D/CUTS_scripts/`
python step01_produce_results.py --config ../../../config/retina_seed2.yaml
```
</details>


<details>
  <summary><b>Stage 2.</b> Results Generation</summary>

#### To generate and save the segmentation using spectral k-means.
```
## Under `src/scripts_analysis`
python generate_kmeans.py --config ../../config/retina_seed2.yaml
```
#### To generate and save the segmentation using diffusion condensation.
```
## Under `src/scripts_analysis`
python generate_diffusion.py --config ../../config/retina_seed2.yaml
```
#### To generate and save the segmentation using baseline methods.
```
## Under `src/scripts_analysis`
python generate_baselines.py --config ../../config/retina_seed2.yaml
```
</details>

<details>
  <summary>Results Plotting</summary>

#### To reproduce the figures in the paper.
There is one single script for this purpose (previously two but we recently merged them): `plot_paper_figure_main.py`.

The `image-idx` argument shall be followed by space-separated index/indices of the images to be plotted.

Without the `--comparison` flag, the CUTS-only results will be plotted.
With the ` --comparison` flag, the side-by-side comparison against other methods will be plotted.

With the ` --grayscale` flag, the input images and reconstructed images will be plotted in grayscale.

With the `--binary` flag, the labels will be binarized using a consistent method described in the paper.

With the `--separate` flag, the labels will be displayed as separate masks. Otherwise they will be overlaid. This flag is altomatically turned on (and cannot be turned off) for multi-class segmentation cases.

```
## Under `src/scripts_analysis`

## For natural images (berkeley), multi-class segmentation.
### Diffusion condensation trajectory.
python plot_paper_figure_main.py --config ../../config/berkeley_seed2.yaml --image-idx 8 22 89
### Segmentation comparison.
python plot_paper_figure_main.py --config ../../config/berkeley_seed2.yaml --image-idx 8 22 89 --comparison --separate

## For medical images with color (retina), binary segmentation.
### Diffusion condensation trajectory.
python plot_paper_figure_main.py --config ../../config/retina_seed2.yaml --image-idx 4 7 18
### Segmentation comparison (overlay).
python plot_paper_figure_main.py --config ../../config/retina_seed2.yaml --image-idx 4 7 18 --comparison --binary
### Segmentation comparison (non-overlay).
python plot_paper_figure_main.py --config ../../config/retina_seed2.yaml --image-idx 4 7 18 --comparison --binary --separate

## For medical images without color (brain ventricles, brain tumor), binary segmentation.
### Diffusion condensation trajectory.
python plot_paper_figure_main.py --config ../../config/brain_ventricles_seed2.yaml --image-idx 35 41 88 --grayscale
### Segmentation comparison (overlay).
python plot_paper_figure_main.py --config ../../config/brain_ventricles_seed2.yaml --image-idx 35 41 88 --grayscale --comparison --binary
### Segmentation comparison (non-overlay).
python plot_paper_figure_main.py --config ../../config/brain_ventricles_seed2.yaml --image-idx 35 41 88 --grayscale --comparison --binary --separate
### Diffusion condensation trajectory.
python plot_paper_figure_main.py --config ../../config/brain_tumor_seed2.yaml --image-idx 1 25 31 --grayscale
### Segmentation comparison (overlay).
python plot_paper_figure_main.py --config ../../config/brain_tumor_seed2.yaml --image-idx 1 25 31 --grayscale --comparison --binary
### Segmentation comparison (non-overlay).
python plot_paper_figure_main.py --config ../../config/brain_tumor_seed2.yaml --image-idx 1 25 31 --grayscale --comparison --binary --separate

## We also have an option to not overlay binary segmentation.
python plot_paper_figure_main.py --config ../../config/retina_seed2.yaml --image-idx 4 7 14 --comparison --binary

```
</details>

<details>
  <summary>Results Analysis</summary>

#### To compute the quantitative metrics (single experiment).
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python run_metrics.py --config ../../config/retina_seed2.yaml
```

#### To compute the quantitative metrics (multiple experiments).
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python run_metrics.py --config ../../config/retina_seed1.yaml ../../config/retina_seed2.yaml ../../config/retina_seed3.yaml
```

</details>

## To train on your custom dataset **with label**.
**NOTE**: Since our method is **completely unsupervised**, the only additional benefit for providing labels are:
1. With labels, we will have meaningful quantitative metrics (dice coefficient, etc.) when you run `run_metrics.py`.
2. With labels, for binary segmentation tasks, our method can predict binary masks in addition to multi-scale segmentations.

The process is largely the same as detailed in the section: **To reproduce the results in the paper** above.

<details>
  <summary>The additional work you need to complete prior to training are</summary>

1. Put your dataset under `src/data/`, similar to the other datasets.
2. Write your custom config file and put it under `config/`, similar to the other config files.
3. Write your custom `Dataset` class in `src/datasets/***.py`, similar to the existing examples.
    - If your dataset is very small (e.g., 50 images), you can refer to `src/datasets/brain_ventricles.py` or `src/datasets/retina.py`, where the data is pre-loaded to the CPU prior to training.
    - If your dataset is rather big, you can refer to `src/datasets/brain_tumor.py`, where the data is loaded on-the-fly during training.
4. Make sure your custom `Dataset` is included in `src/data_utils/prepare_datasets.py`, both in the import section on the top of the page, and inside the `prepare_dataset` function, alongside the lines such as `dataset = Retina(base_path=config.dataset_path)`.
5. Currently, most of our example `Dataset` classes expect matching names between the image and the label. If your data is organized differently, please be mindful you need to change your logic in your class. (Credit to [DerrickGuu](https://github.com/ChenLiu-1996/CUTS/issues/13#issuecomment-2192022353))
</details>

Other than that, you can use the pipeline as usual.

## To train on your custom dataset **without label**.

The process is largely the same as detailed in the section: **To reproduce the results in the paper** above.

<details>
  <summary>The additional work you need to complete prior to training are</summary>

1. Put your dataset under `src/data/`, similar to the other datasets.
2. Write your custom config file and put it under `config/`, similar to the other config files. Please note that, just like `example_dataset_without_label_seed1.yaml`, you shall specify the additional field `no_label: True`.
3. Write your custom `Dataset` class in `src/datasets/***.py`, similar to the existing examples.
    - If your dataset is very small (e.g., 50 images), you can refer to `src/datasets/brain_ventricles.py` or `src/datasets/retina.py`, where the data is pre-loaded to the CPU prior to training.
    - If your dataset is rather big, you can refer to `src/datasets/brain_tumor.py`, where the data is loaded on-the-fly during training.
    - However, you need to pay attention that, since your custom dataset does not have labels, you shall refer to `src/datasets/example_dataset_without_label.py` to see how you need to use an `np.nan` as a placeholder for the non-existent labels inside the `__getitem__` method.
4. Make sure your custom `Dataset` is included in `src/data_utils/prepare_datasets.py`, both in the import section on the top of the page, and inside the `prepare_dataset` function, alongside the lines such as `dataset = ExampleDatasetWithoutLabel(base_path=config.dataset_path)`.
5. Currently, most of our example `Dataset` classes expect matching names between the image and the label. If your data is organized differently, please be mindful you need to change your logic in your class. (Credit to [DerrickGuu](https://github.com/ChenLiu-1996/CUTS/issues/13#issuecomment-2192022353))
</details>

Other than that, you can use the pipeline as usual.

Be mindful though: when you run `generate_kmeans.py`, the script will still print out dice scores for each image. The values shall be very close to zero (on the order of 1e-16). This does not mean the segmentation is bad. This only means the ground truth label is not provided. The dice score is computed against a placeholding all-zero label, with a very tiny numerical stability term.

**SPECIAL NOTE**: The outcome of this pipeline will be the multi-scale segmentations as a result of diffusion condensation. No binary mask will be generated. **If you really want the model to generate binary masks in addition to the multi-scale segmentations, what you can do is to provide a set of pseudo-labels** as follows:
1. Instead of segmenting the desired region-of-interest carefully as in regular labels, you just casually circle/square/whatever a typical enough subregion inside the region-of-interest, and mark them as 1's whereas the backgrounds as 0's. You can do this with any labeling tool you like.
2. Then, you provide these casual pseudo-labels as if they were real labels. Put them in the correct folder under `src/data/` and load them as if they were real labels in your `src/datasets/***.py`.
3. Follow the pipeline as in the previous section: **To train on your custom dataset with label**.
4. In this way, binary masks will also be generated. Please be careful though: the quantitative metrics will not be accurate, as you are providing pseudo-labels instead of accurate real labels.

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name cuts pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate cuts
conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda
python -m pip install -U phate
python -m pip install multiscale-phate
python -m pip install git+https://github.com/KrishnaswamyLab/CATCH
python -m pip install opencv-python-headless
python -m pip install sewar
python -m pip install monai
python -m pip install nibabel

# (Optional) For STEGO
python -m pip install omegaconf
python -m pip install wget
python -m pip install torchmetrics
python -m pip install tensorboard
python -m pip install pytorch-lightning==1.9
python -m pip install azureml
python -m pip install azureml.core

# (Optional) For SAM
python -m pip install git+https://github.com/facebookresearch/segment-anything.git

# (Optional) For MedSAM
python -m pip install git+https://github.com/bowang-lab/MedSAM.git

# (Optional) For SAM-Med2D
python -m pip install albumentations
python -m pip install scikit-learn==1.1.3  # need to downgrade to 1.1.3
```
Installation usually takes between 20 minutes and 1 hour on a normal desktop computer.

## DEBUG Notes
<details>
  <summary>Regarding "dead lock" (e.g., never-ending repeated `Time out!`) when generating results.</summary>

On our YCRC server, sometimes we need to run
```
export MKL_THREADING_LAYER=GNU
```
before running some of the code code to minimize the risk of dead lock. For details, see https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md.

**UPDATE Dec 26, 2022**: I finally wrote a workaround to avoid running the script over and over again from the first incomplete file whenever a deadlock is hit (which is a total waste of human efforts)! The method is simple: in `generate_kmeans.py`, if we turn on the `-r` flag, we will outsource the kmeans computation and numpy saving to a helper file `helper_generate_kmeans.py`, and we kill and restart the helper whenever a deadlock causes the process to timeout. **However**, on our YCRC server, you may **still** need to run the command `export MKL_THREADING_LAYER=GNU` to minimize risk of dead lock.

</details>

<details>
  <summary>Regarding `zsh bus error`.</summary>

If you encounter `zsh bus error` while running some of the python scripts, for example, `generate_kmeans.py` or `generate_diffusion.py`, it is very likely that the program requires more RAM than available. On our YCRC, the solution is to request more RAM for the job.
</details>


## Acknowledgements

For the comparison against other methods, we use the official implementations from the following repositories:
- [**DFC**, *IEEE TIP 2020*: Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip)
- [**STEGO**, *ICLR 2022*: Unsupervised Semantic Segmentation by Distilling Feature Correspondences](https://github.com/mhamilton723/STEGO)
- [**SAM**, *ICCV 2023* (Meta AI Research): Segment Anything](https://github.com/facebookresearch/segment-anything)
- [**SAM**, *Medical Image Analysis 2024*: Segment Anything Model for Medical Image Analysis: an Experimental Study](https://github.com/mazurowski-lab/segment-anything-medical-evaluation)
- [**SAM-Med2D**, *ArXiv*: SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D)
- [**MedSAM**, *Nature Communications 2024*: Segment anything in medical images](https://github.com/bowang-lab/MedSAM)
