export PYTHONPATH=.:$PYTHONPATH
CONFIG=$1
python train.py --config_name $CONFIG