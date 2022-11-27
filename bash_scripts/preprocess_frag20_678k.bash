#!/bin/bash

export PYTHONPATH=.
python scripts/frag20_sol_all.py
python scripts/csd20_sol_all.py
python scripts/conf20_sol_all.py

python scripts/prepare_frag20-678k.py
