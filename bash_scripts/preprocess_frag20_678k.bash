#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

### the following 3 commands support multi-processing. About 4 hours on 12 CPUs and 30GB of memory
python scripts/frag20_sol_all.py 
python scripts/csd20_sol_all.py 
python scripts/conf20_sol_all.py 

### The following command does not support multi-processing. About 90 minutes on one CPU and 35GB of memory
python scripts/prepare_frag20-678k.py
