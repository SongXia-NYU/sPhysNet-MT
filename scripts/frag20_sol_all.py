import multiprocessing
import os.path as osp
from multiprocessing import Pool
import os

import pandas as pd
import tqdm
from tqdm.contrib.concurrent import process_map

from utils.frag20_sol_single import frag20_csd20_sol_single, RAW_DATA_ROOT, TEMP_DATA_ROOT, get_debug_arg
from utils.utils_functions import solv_num_workers


def frag20_sol_single_wrapper(df):
    geometry = "mmff"
    sample_id = df.iloc[0]["FileHandle"]
    out_path = osp.join(TEMP_DATA_ROOT, "frag20", geometry, f"{sample_id}.pyg")
    try:
        frag20_csd20_sol_single(sample_id, None, geometry, out_path, df)
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")


def frag20_sol_all():
    args = get_debug_arg()

    df = pd.read_csv(f"{RAW_DATA_ROOT}/frag20_solvation_with_fl.csv")
    df = df[df["SourceFile"] != "CCDC"]

    print("splitting dfs...")
    if args.debug:
        dfs = [df.iloc[[i]] for i in range(20)]
    else:
        dfs = [df.iloc[[i]] for i in range(df.shape[0])]
    print("splitting done..")

    n_cpu_avail, n_cpu, num_workers = solv_num_workers()
    print(f"Number of available CPUs: {num_workers}")

    os.makedirs(osp.join(TEMP_DATA_ROOT, "frag20", "mmff"), exist_ok=True)

    process_map(frag20_sol_single_wrapper, dfs, total=len(dfs), chunksize=20, max_workers=num_workers, desc="proceesing Frag20")


if __name__ == '__main__':
    frag20_sol_all()

