export PYTHONPATH=.:$PYTHONPATH
CONFIG=$1
python active_learning.py --config_name $CONFIG --fixed_train --fixed_valid --action_n_heavy "" --metric ENSEMBLE