cd ..
export PYTHONPATH=.:$PYTHONPATH

python test.py --folder_names '../raw_data/frag20-sol-finals/exp_frag20sol_012_active_external_plati20_ALL_2022-05-01_112820/exp_*_run_*' --config_folder '../raw_data/frag20-sol-finals/exp_frag20sol_012_active_external_plati20_ALL_2022-05-01_112820' --no_runtime_split