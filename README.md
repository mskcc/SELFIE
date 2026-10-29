# SELFIE

"SELFIE: Self-supervised learning for fast dynamic golden-angle radial MRI" is a self-supervised deep learning framework for reconstructing dynamic golden-angle radial MRI data. It exploits the flexible temporal resolution property of golden-angle radial acquisitions and inherent sparsity of dynamic imaging to self-supervise reconstruction network training. By enforcing sparsity and consistency between representations at different temporal resolutions, network training is performed using only data in image space. SELFIE was evaluated on gynecological dynamic contrast-enhanced (DCE)-MRI and motion-resolved free-breathing abdominal MRI against compressed sensing (CS) and supervised deep learning (SDL). 

You can find code to train and run the deep learning-based SELFIE and SDL experiments. It as tested on a 20.04 Ubuntu machine with 4 NVIDIA A100 GPU nodes (CUDA 11.4).\
<mark> Important: Please read the Terms of Use License before using this repository! <mark>

## Installation

1. clone the SELFIE repository
2. create a new virtual environemnt with python>= 3.10 and activate it (e.g., ```conda create -n SELFIE python=3.10```)
3. cd into the SELFIE repository
4. install the SELFIE package via:\
```pip install .```
5. if you want to use GPU, install torch via:\
```pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 ```
6. Make sure you have the path to the SELFIE repository added to your ```~/.bashrc``` (e.g., ```export PYTHONPATH="${PYTHONPATH}:/home/username/SELFIE"```)
7. For validation, you need to clone https://github.com/Bayer-Group/mr-image-metrics repository. You also need to delete the ```mr-image-metrics/medimetrics/metrics.__init__.py```, as we don't want to include all metrics and packages. Make sure you add the path to mr-image-metrics to your ```~/.bashrc``` as well. 
8. Make sure you add the required paths to the data/models to your ```~/.bashrc```. See the following list:\
    ```export DATA_DIR_RECOS=/otazolab_ess/Schellenberg/ProcessedData/Data_example/``` \
    ```export DATA_DIR_EXP=/otazolab_ess/Schellenberg/ProcessedData/ProcessedData_example/``` \
    ```export RESULTS_DIR=/otazolab_ess/Schellenberg/ProcessedData/Networks_results/``` \
    ```export RESULTS_ANALYSIS_DIR=/otazolab_ess/Schellenberg/ProcessedData/Networks_results_analysis/``` \
    ```export MASK_DIR=/otazolab_ess/Schellenberg/ProcessedData/Data_example/```\

## Usage

You can use this repository to preprocess your datasets in order to train or apply Movienet or SELFIE.

### Dataset requirements

The data must be located in one folder in ```*.h5py``` or ```*.mat``` format. It must include the following files:
- complex-valued, density-compensated, coil-combined NUFFTs named ```rnufft```
- complex-valued, coil-combined GRASP reconstructions named ```x```
- complex-valued, density-compensated, coil-wise NUFFTs named ```coilwise_rnufft```

Three dummy slices (Data_example) of that structure can be found under: 10.5281/zenodo.18664249. Note, that the only purpose of these dummy slices is to show the structure of the data. The slices show the shepp logan phantom that was artificially changed to complex data and each slice shows one global contrast enhancement (generated using the extended Tofts model). This means, the dummy data doesn't show spatially varying contrast enhancements and also no phase information. Results on these images won't be reasonable. 

## Preprocessing

Before training the models or running inference, the data needs to be split and the normalization factors need to be computed. For each experiment (gynecological DCE-MRI and motion-resolved free-breathing abdominal MRI), the reconstruction files are split via ```/data/split_data_example.py```. Note, that you need to specify the studies for training/validation/test in the script, if you want to work with your data. the data used here was reconstructed slice-wise and each slice was of shape 400x400. The split scripts load the NUFFT, coil-wise NUFFT, and GRASP files, crops it to 256x256 pixels, and splits it into training, validation, and test datasets. 

Additionally, you need to run ```/data/get_norm.py```, which calculates and saves the 90th percentile of each NUFFT slice averaged across the dynamic dimension.

You can plot some example files with ```/visualization/plot_example.py```. Also here you need to make adaptations if you work with your data.

### Training

1. Edit the config file under ```/config/config_example.yaml```.
2. ```python3 /training/train_model.py```

### Inference

1. Edit the config file under ```/config/config_test.yaml```.
2. ```python3 /inference/infer_model.py```

### Analysis

For analysis of the results, please run ```/visualization/analysis_example.py```. Please adapt the script according to your data. In general, it loads the results and creates *.gifs for selected cases and slices. It also loads the example ROI file to plot the mean contrast enhancement there. For both experiments, you can choose to calculate metrics and to plot them. 

## Contributing

Feel free to reach out, if you face any issues.

## Citation
Schellenberg, M, Mekhanik A, Murray V, Argiriadi P, Lakhman Y, Levin G, Akin O, Otazo R. "SELFIE: Self-supervised learning for fast dynamic golden-angle radial MRI”, NMR in Biomedicine, 2026. 
