#!/bin/bash

conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch -y
conda install pyg -c pyg -y
conda install -c conda-forge rdkit==2021.09.5 -y
pip install treelib
pip install ase