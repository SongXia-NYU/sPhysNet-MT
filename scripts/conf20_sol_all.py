import multiprocessing
import os
import os.path as osp
from multiprocessing import Pool

import pandas as pd
import tqdm

from utils.frag20_sol_single import sdf2pyg_single

root = "/ext3/Conf20_sol"


def conf20_sol_single_wrapper(args):
    df, geometry = args
    sample_id = df.iloc[0]["sample_id"]
    if geometry == "qm":
        sdf = osp.join(root, "Conf20_process", "qm_sdf", f"{sample_id}.qm.sdf")
    else:
        sdf = osp.join(root, "Conf20_process", "mmff_sdfs", f"{sample_id}.sdf")
    out_path = osp.join(root, "single", geometry, f"{sample_id}.pyg")
    try:
        sdf2pyg_single(sdf, df, out_path)
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")


def conf20_sol_all():
    df = pd.read_csv(osp.join(root, "summary", "gauss_sol.csv"))
    df = df.dropna()

    print("splitting dfs......")
    dfs = [df.iloc[[i]] for i in range(df.shape[0])]
    print("splitting done.....")

    n_cpu = multiprocessing.cpu_count()
    print(f"Number of CPUs detected: {n_cpu}")

    with Pool(n_cpu) as p:
        for geometry in ["qm", "mmff"]:
            os.makedirs(osp.join(root, "single", geometry), exist_ok=True)
            args = [(df, geometry) for df in dfs]
            for _ in tqdm.tqdm(p.imap_unordered(conf20_sol_single_wrapper, args), total=len(dfs), desc=geometry):
                pass


if __name__ == '__main__':
    conf20_sol_all()
