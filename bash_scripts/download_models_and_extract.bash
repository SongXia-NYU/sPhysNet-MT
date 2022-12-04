#!/bin/bash

mkdir -p ./data/models
cd ./data/models

wget https://yzhang.hpc.nyu.edu/IMA/Datasets/sPhysNet-MT_cal_trained_models.tar.gz
wget https://yzhang.hpc.nyu.edu/IMA/Datasets/sPhysNet-MT_exp_trained_models.tar.gz

tar xvf sPhysNet-MT_cal_trained_models.tar.gz

## extract models trained on the experimental dataset. Since 50 random splits are trained, we only extract the first split for demostration
tar xvf sPhysNet-MT_exp_trained_models.tar.gz --wildcards exp_ultimate_freeSolv_13_RDrun_2022-05-20_100307__201005/exp_ultimate_freeSolv_13_active_ALL_2022-05-20_100309