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

Download the raw data (labels and geometry files) from [our website](https://yzhang.hpc.nyu.edu/IMA/). The downloaded data will be stored in `./data/raw`

`bash bash_scripts/download_data_and_extract.bash`

Preprocess the calculated dataset (Frag20-solv-678k). The following command takes around ??? hours with 12 CPUs and 30GB of memory.

`bash bash_scripts/preprocess_frag20_678k.bash`

Preprocess the experimental dataset (FreeSolv-PHYSPROP-14k). The following command takes around 6 minutes with 2 CPUs.

`bash bash_scripts/preprocess_freesolv_physprop.bash`

### 2. Train sPhysNet-MT on the calculated dataset (Frag20-solv-678k)

Train a sPhysNet-MT from scratch. It is recommended to use 30GB memory to train on Frag20-solv-678k.

`bash bash_scripts/train_from_config.bash configs/config-frag20sol.txt`

Train a ensemble of 5 models from scratch on the calculated dataset.

`bash bash_scripts/train_ens_from_config.bash configs/config-frag20sol.txt`

### 2. Train sPhysNet-MT on the experimental dataset (FreeSolv-PHYSPROP-14k)

Fine-tune the pretrained model on FreeSolv-PHYSPROP-14k using 50 random splits and 5 ensembles.

`bash bash_scripts/train_rd_split_from_config.bash configs/config-freesolv_physprop.txt`

## Model Evaluation

After training, you will find the trained folder in the current directory: `./exp_*_run_*` (single run), `./exp_*_active_ALL_*` (ensemble) or `./exp_*_RDrun_*` (random split). Those folders contain all the information about the model training as well as the model with the lowest evaluation loss. To evaluate the performance on the test set:

`bash bash_scripts/test.bash $TRAINED_FOLDER`

Replace `$TRAINED_FOLDER` with the folder path you actually get, for example:

`bash bash_scripts/test.bash exp_frag20sol_run_2022-11-26_205046__623751`
