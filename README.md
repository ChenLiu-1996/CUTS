# CUTS: A Fully Unsupervised Framework for Medical Image Segmentation
### Krishnaswamy Lab, Yale University


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
```

## Usage
### Activate environment.
```
conda activate $OUR_CONDA_ENV
```
### Training and Testing
#### To train a model.
```
## Under $CUTS_ROOT/src
python main.py --mode train --config ../config/$YAML_FILE.yaml
```
#### To test a model (automatically done during `train` mode).
```
## Under $CUTS_ROOT/src
python main.py --mode test --config ../config/$YAML_FILE.yaml
```

### Results Generation.
#### To generate and save the segmentation using spectral k-means.
```
## Under $CUTS_ROOT/src/scripts_analysis
python generate_kmeans.py --config ../../config/$YAML_FILE.yaml
```
#### To generate and save the segmentation using diffusion condensation.
```
## Under $CUTS_ROOT/src/scripts_analysis
python generate_diffusion.py --config ../../config/$YAML_FILE.yaml
```
#### To generate and save the segmentation using baseline methods.
```
## Under $CUTS_ROOT/src/scripts_analysis
python generate_baselines.py
```

### Results Plotting.
#### *To reproduce the figures in the paper.*
```
## Under $CUTS_ROOT/src/scripts_analysis

## For natural images (berkeley)
python plot_paper_figure_natural.py --config ../../config/$YAML_FILE.yaml --image-idx $IMAGE_IDX
python plot_paper_figure_natural.py --config ../../config/$YAML_FILE.yaml --image-idx $IMAGE_IDX --comparison

## For medical images (retina, brain)
python plot_paper_figure_medical.py --config ../../config/$YAML_FILE.yaml --image-idx $IMAGE_IDX
python plot_paper_figure_medical.py --config ../../config/$YAML_FILE.yaml --image-idx $IMAGE_IDX --comparison
```
#### To plot the segmentation results using spectral k-means.
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python plot_kmeans.py --config ../../config/$YAML_FILE.yaml
```
#### To plot the segmentation results using diffusion condensation.
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python plot_diffusion.py --config ../../config/$YAML_FILE.yaml
```

### Results Analysis.
#### To compute the quantitative metrics.
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python run_metrics.py --config ../../config/$YAML_FILE.yaml
```

### Special NOTE
On our YCRC server, sometimes we need to run
```
export MKL_THREADING_LAYER=GNU
```
before running some results generation/plotting/analysis code to avoid dead lock.

For details, see https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md.