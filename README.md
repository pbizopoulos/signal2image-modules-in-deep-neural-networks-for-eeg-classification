[![citation](http://img.shields.io/badge/Citation-0091FF.svg)](https://scholar.google.com/scholar?q=Signal2Image%20Modules%20in%20Deep%20Neural%20Networks%20for%20EEG%20Classification.%20arXiv%202019)
[![arXiv](http://img.shields.io/badge/eess.SP-arXiv%3A1904.13216-B31B1B.svg)](https://arxiv.org/abs/1904.13216)

# Signal2Image Modules in Deep Neural Networks for EEG Classification
This repository contains the code that generates the results of the paper **Signal2Image Modules in Deep Neural Networks for EEG Classification** presented at EMBC 2019.

## Requirements
- docker
- grep
- make
- nvidia-container-toolkit [required only for using cuda]
- python3-venv

## Instructions
1. `git clone https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification`
2. `cd signal2image-modules-in-deep-neural-networks-for-eeg-classification`
3. `sudo systemctl start docker`
4. `make venv-verify` or `make docker-verify`
