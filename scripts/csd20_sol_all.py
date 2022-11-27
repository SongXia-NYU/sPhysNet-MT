import multiprocessing
import os
import os.path as osp
from multiprocessing import Pool

import pandas as pd
import tqdm

from utils.frag20_sol_single import frag20_csd20_sol_single

root = "/ext3/CSD20_sol"


def csd20_sol_single_wrapper(args):
    df, geometry = args
    sample_id = df.iloc[0]["FileHandle"]
    out_path = osp.join(root, "single", geometry, f"{sample_id}.pyg")
    try:
        frag20_csd20_sol_single(sample_id, None, geometry, out_path, df)
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")


def csd20_sol_all():
    df = pd.read_csv("/ext3/CSD20_sol/frag20_solvation_with_split_fl.csv")
    df = df[df["SourceFile"] == "CCDC"]

    print("splitting dfs...")
    dfs = [df.iloc[[i]] for i in range(df.shape[0])]
    print("splitting done..")

    n_cpu = multiprocessing.cpu_count()
    print(f"Number of CPUs detected: {n_cpu}")

    with Pool(n_cpu) as p:
        for geometry in ["qm", "mmff"]:
            os.makedirs(osp.join(root, "single", geometry), exist_ok=True)
            args = [(df, geometry) for df in dfs]
            for _ in tqdm.tqdm(p.imap_unordered(csd20_sol_single_wrapper, args), total=len(dfs), desc=geometry):
                pass


if __name__ == '__main__':
    csd20_sol_all()
