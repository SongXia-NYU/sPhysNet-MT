import multiprocessing
import os
import os.path as osp
from multiprocessing import Pool

import pandas as pd
from tqdm.contrib.concurrent import process_map

from utils.frag20_sol_single import RAW_DATA_ROOT, TEMP_DATA_ROOT, frag20_csd20_sol_single, get_debug_arg
from utils.utils_functions import solv_num_workers


def csd20_sol_single_wrapper(args):
    df, geometry = args
    sample_id = df.iloc[0]["FileHandle"]
    out_path = osp.join(TEMP_DATA_ROOT, "csd20", geometry, f"{sample_id}.pyg")
    try:
        frag20_csd20_sol_single(sample_id, None, geometry, out_path, df)
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")


def csd20_sol_all():
    args = get_debug_arg()
    
    df = pd.read_csv(f"{RAW_DATA_ROOT}/frag20_solvation_with_fl.csv")
    df = df[df["SourceFile"] == "CCDC"]

    print("splitting dfs...")
    if args.debug:
        dfs = [df.iloc[[i]] for i in range(20)]
    else:
        dfs = [df.iloc[[i]] for i in range(df.shape[0])]
    print("splitting done..")

    n_cpu_avail, n_cpu, num_workers = solv_num_workers()
    print(f"Number of available CPUs: {num_workers}")

    for geometry in ["mmff"]:
        os.makedirs(osp.join(TEMP_DATA_ROOT, "csd20", geometry), exist_ok=True)
        args = [(df, geometry) for df in dfs]
        process_map(csd20_sol_single_wrapper, args, total=len(dfs), chunksize=20, max_workers=num_workers, desc="proceesing CSD20")


if __name__ == '__main__':
    csd20_sol_all()
