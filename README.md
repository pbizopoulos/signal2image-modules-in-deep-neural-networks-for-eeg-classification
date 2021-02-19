[![arXiv](http://img.shields.io/badge/eess.SP-arXiv%3A1904.13216-B31B1B.svg)](https://arxiv.org/abs/1904.13216)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Signal2Image%20Modules%20in%20Deep%20Neural%20Networks%20for%20EEG%20Classification.%20arXiv%202019)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/a-makefile-for-developing-containerized-latex-technical-documents-template)
[![test-draft-version-document-reproducibility](https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/workflows/test-draft-version-document-reproducibility/badge.svg)](https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/actions?query=workflow%3Atest-draft-version-document-reproducibility)

# Signal2Image Modules in Deep Neural Networks for EEG Classification
This repository contains the code that generates **Signal2Image Modules in Deep Neural Networks for EEG Classification** presented at EMBC 2019.

## Requirements
- [Docker](https://docs.docker.com/get-docker/)
    - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (Optional)
- [Make](https://www.gnu.org/software/make/)

## Instructions
1. `git clone https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification`
2. `cd signal2image-modules-in-deep-neural-networks-for-eeg-classification/`
3. `sudo systemctl start docker`
4. make options
    - `make` # Generate the draft (fast) version document.
    - `make VERSION=--full` # Generate the full (slow) version document.
    - `make clean` # Remove the tmp/ directory.
