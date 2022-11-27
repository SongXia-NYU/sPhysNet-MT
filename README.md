# Multi-task Deep Ensemble Prediction of Molecular Energetics in Solution: From Quantum Mechanics to Experimental Properties
![](figures/toc.png)

## Environment Setup
1. If you are using Linux and Conda

    Create a new conda environment with python 3.8:

    `conda create -n sphysnet-mt python==3.8 -y`

    `conda activate sphysnet-mt`

    Install the required packages:

    `bash bash_scripts/install_env_linux.bash`
2. If you are on other systems or are using other package managers, please install the following packages:

- [PyTorch](https://pytorch.org/)
- [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Rdkit](https://www.rdkit.org/docs/Install.html)
- [TreeLib](https://pypi.org/project/treelib/)
- [ASE](https://pypi.org/project/ase/)

## Run Prediction

WIP...

## Model Training

### 1. Dataset preprocess

The follow script will download the raw geometry(SDF) files and labels from [our website](https://yzhang.hpc.nyu.edu/IMA/). The downloaded data will be stored in `./data/raw`

`bash bash_scripts/download_data_and_extract.bash`
