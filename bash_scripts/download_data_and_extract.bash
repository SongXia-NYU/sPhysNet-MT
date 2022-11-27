#!/bin/bash
mkdir -p ./data/raw
cd ./data/raw

## CSD20 geometry files
wget https://yzhang.hpc.nyu.edu/IMA/Datasets/CSD20.tar.bz2
tar xvf CSD20.tar.bz2

## Frag20 geometry files
wget https://yzhang.hpc.nyu.edu/IMA/Datasets/Frag20.tar.bz2
tar xvf Frag20.tar.bz2