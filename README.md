# DAFD_EFGM

## Data Augmentation with Flow-based Generative Model

Code accompanying the paper "Data Augmentation on Fault Diagnosis of Wind Turbine Gearboxes with an Enhanced Flow-based Generative Model" by Authors (Ready to be submitted for publication).

-  Tensorflow 1.15.0 implementation
-  Inspired by Diederik P. Kingma $et$ $al$. [Glow: Generative flow with invertible 1x1 convolutions](https://proceedings.neurips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf)
- Some code is based on Keras implement of flow-based models: https://github.com/bojone/flow

## Requirements

- python 3.7.0
- Tensorflow == 1.15.0
- Keras == 2.3.1

Note: All experiment were excecuted on an Nvidia RTX 3060 Laptop GPU.

## Main file discription
* `--glow_feature`: Main code of the Glow model
* `--random_forest`: Main code of the RF classifier
* `--WPT`: Main code of the WPT feature extraction

## Implementation details
- Note that users should change the directory to successfully run this code.
- Hyperparameter settings: Adam optimizer is used with learning rate of `1e-4` in Glow model ;The batch size is `32`, total iteration for Glow model is `50000`. For the random forest, 500 nodes were chosen with their default setting.  
