export PYTHONPATH=.:$PYTHONPATH
CONFIG=$1
python train_rd_split.py --config_name $CONFIG