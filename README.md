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
<details>
  <summary>Activate environment</summary>

```
conda activate $OUR_CONDA_ENV
```
</details>

<details>
  <summary>Training and Testing</summary>

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
  <summary>Results Generation</summary>

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
#### To plot the segmentation results using spectral k-means (optional).
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python plot_kmeans.py --config ../../config/$CONFIG_FILE.yaml
```
#### To plot the segmentation results using diffusion condensation (optional).
Assuming segmentation results have already been generated and saved.
```
## Under $CUTS_ROOT/src/scripts_analysis
python plot_diffusion.py --config ../../config/$CONFIG_FILE.yaml
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

### Special NOTES
<details>
  <summary>1. Regarding occasional "deadlock" when generating/plotting results.</summary>

On our YCRC server, sometimes we need to run
```
export MKL_THREADING_LAYER=GNU
```
before running some results generation/plotting/analysis code to avoid dead lock.

For details, see https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md.
</details>

<details>
  <summary>2. Regarding `latent_evaluator` and reported dice coeff in `test` mode.</summary>

You may notice something called `latent_evaluator` in `main.py`.

At first, I wrote it to evaluate the model during test time. However, eventually I decided to off-source this kind of jobs to separate scripts under the `scripts_analysis` folder, and rather use it as a numpy results saver. As of now, I haven't changed `latent_evaluator` to `results_saver`, but I may do that at some point.

Along the same lines, you could feel free to leave the following hyperparameters as-is in your config yaml:
```
segmentation_paradigm: 'kmeans_point'
test_metric: None
```
In this case, the `latent_evaluator` will indeed act as a numpy results saver. Since no evaluation is actually performed, the logging will say: **dice coeff: nan ± nan, which is not a bug and you shall not be scared of it**.

</details>

