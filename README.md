![test-reproducible-build](https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/workflows/test-reproducible-build/badge.svg)
[![arXiv](http://img.shields.io/badge/eess.SP-arXiv%3A1904.13216-B31B1B.svg)](https://arxiv.org/abs/1904.13216)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Signal2Image%20Modules%20in%20Deep%20Neural%20Networks%20for%20EEG%20Classification.%20arXiv%202019)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/cookiecutter-reproducible-builds-for-computational-research-papers)

# Signal2Image Modules in Deep Neural Networks for EEG Classification
This repository contains the code that generates the results of the paper **Signal2Image Modules in Deep Neural Networks for EEG Classification** presented at EMBC 2019.

## Requirements
- UNIX utilities (cmp, cp, echo, rm, touch)
- [docker](https://docs.docker.com/get-docker/)
- make
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (required only when using CUDA)

## Instructions
1. `git clone https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification`
2. `cd signal2image-modules-in-deep-neural-networks-for-eeg-classification`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate pdf.
    * `make ARGS=--full` # Generate full pdf.
    * `make clean`       # Remove cache, results directories and tex auxiliary files.
