# Signal2Image Modules in Deep Neural Networks for EEG Classification
This repository contains the code that generates the results of the paper **Signal2Image Modules in Deep Neural Networks for EEG Classification** presented at EMBC 2019.

ArXiv link: <https://arxiv.org/abs/1904.13216>

# Instructions
The syntax of the `make` command is as follows:

`make [docker] [ARGS="[--full] [--gpu]"]`

where `[...]` denotes an optional argument.

For example you can choose one of the following:
- `make`
	- Requires local installation of requirements.txt and texlive-full.
	- Takes ~2 minutes and populates the figures and table.
- `make ARGS="--full --gpu"`
	- Requires local installation of requirements.txt and texlive-full.
	- Takes a week on an NVIDIA Titan X.
- `make docker`
	- Requires local installation of docker.
	- Takes ~2 minutes.
- `make docker ARGS="--full --gpu"`
	- Requires local installation of nvidia-container-toolkit.
	- Takes a week on an NVIDIA Titan X.
- `make clean`
	- Restores the repo in its initial state by removing all figures, tables and downloaded datasets.

# Citation
If you use this repository cite the following:
```
@inproceedings{bizopoulos2019signal2image,
	title={Signal2image modules in deep neural networks for eeg classification},
	author={Bizopoulos, Paschalis and Lambrou, George I and Koutsouris, Dimitrios},
	booktitle={2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
	pages={702--705},
	year={2019},
	organization={IEEE}
}
```
