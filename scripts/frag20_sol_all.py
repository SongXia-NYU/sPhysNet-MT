import multiprocessing
import os.path as osp
from multiprocessing import Pool

import pandas as pd
import tqdm

from utils.frag20_sol_single import frag20_csd20_sol_single

root = "/ext3/Frag20-Sol"


def frag20_sol_single_wrapper(df):
    geometry = "mmff"
    sample_id = df.iloc[0]["FileHandle"]
    out_path = osp.join(root, "single", geometry, f"{sample_id}.pyg")
    try:
        frag20_csd20_sol_single(sample_id, None, geometry, out_path, df)
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")


def frag20_sol_all():

    df = pd.read_csv("/ext3/Frag20-Sol/frag20_solvation_with_split_fl.csv")

    print("splitting dfs...")
    dfs = [df.iloc[[i]] for i in range(df.shape[0])]
    print("splitting done..")

    n_cpu = multiprocessing.cpu_count()
    print(f"Number of CPUs detected: {n_cpu}")

    with Pool(n_cpu) as p:
        for _ in tqdm.tqdm(p.imap_unordered(frag20_sol_single_wrapper, dfs), total=len(dfs)):
            pass


if __name__ == '__main__':
    frag20_sol_all()

