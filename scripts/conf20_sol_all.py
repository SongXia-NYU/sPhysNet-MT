import multiprocessing
import os
import os.path as osp
from multiprocessing import Pool

import pandas as pd
from tqdm.contrib.concurrent import process_map

from utils.frag20_sol_single import RAW_DATA_ROOT, TEMP_DATA_ROOT, get_debug_arg, sdf2pyg_single
from utils.utils_functions import solv_num_workers


def conf20_sol_single_wrapper(args):
    df, geometry = args
    sample_id = df.iloc[0]["sample_id"]
    if geometry == "qm":
        sdf = osp.join(RAW_DATA_ROOT, "Conf20_sol", "qm_sdf", f"{sample_id}.qm.sdf")
    else:
        sdf = osp.join(RAW_DATA_ROOT, "Conf20_sol", "mmff_sdfs", f"{sample_id}.sdf")
    out_path = osp.join(TEMP_DATA_ROOT, "conf20", geometry, f"{sample_id}.pyg")
    try:
        sdf2pyg_single(sdf, df, out_path)
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")


def conf20_sol_all():
    args = get_debug_arg()

    df = pd.read_csv(osp.join(RAW_DATA_ROOT, "Conf20_sol", "summary", "gauss_sol.csv"))
    df = df.dropna()

    print("splitting dfs......")
    if args.debug:
        dfs = [df.iloc[[i]] for i in range(20)]
    else:
        dfs = [df.iloc[[i]] for i in range(df.shape[0])]
    print("splitting done.....")

    n_cpu_avail, n_cpu, num_workers = solv_num_workers()
    print(f"Number of available CPUs: {num_workers}")

    for geometry in ["mmff"]:
        os.makedirs(osp.join(TEMP_DATA_ROOT, "conf20", geometry), exist_ok=True)
        args = [(df, geometry) for df in dfs]
        process_map(conf20_sol_single_wrapper, args, total=len(dfs), chunksize=20, max_workers=num_workers, desc="proceesing Conf20")


if __name__ == '__main__':
    conf20_sol_all()
