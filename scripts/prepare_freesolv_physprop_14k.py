from glob import glob
import os
import os.path as osp
from typing import Set
from tqdm.contrib.concurrent import process_map

from utils.frag20_sol_single import RAW_DATA_ROOT, TEMP_DATA_ROOT, PROCESSED_DATA_ROOT
from utils.sdf2pyg_single import sdf2pyg_single
from utils.utils_functions import solv_num_workers

import pandas as pd
import os.path as osp

import torch
import torch_geometric.data

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15

PROPS = ["gasEnergy", "watEnergy", "octEnergy", "CalcSol", "CalcOct", "watOct"]

def sdf2pyg_wrapper(arg):
    this_sdf, df, out_path = arg
    try:
        sdf2pyg_single(this_sdf, df, out_path)
    except AttributeError as e:
        print(f"Error processing {arg}")
        print(e)

def mp_proccess():
    os.makedirs(osp.join(TEMP_DATA_ROOT, "freesolv"), exist_ok=True)
    os.makedirs(osp.join(TEMP_DATA_ROOT, "physprop"), exist_ok=True)

    freesolv_sdfs = glob(osp.join(RAW_DATA_ROOT, "FreeSolv_PHYSPROP", "freesolv_mmff_sdfs", "*.mmff.sdf"))
    physprop_sdfs = glob(osp.join(RAW_DATA_ROOT, "FreeSolv_PHYSPROP", "openchem_logP_mmff_sdfs", "*.mmff.sdf"))

    mp_args = []
    for sdf in freesolv_sdfs:
        out_pyg = osp.join(TEMP_DATA_ROOT, "freesolv", osp.basename(sdf).split(".mmff.sdf")[0]+".pyg")
        mp_args.append((sdf, None, out_pyg))

    for sdf in physprop_sdfs:
        out_pyg = osp.join(TEMP_DATA_ROOT, "physprop", osp.basename(sdf).split(".mmff.sdf")[0]+".pyg")
        mp_args.append((sdf, None, out_pyg))

    n_cpu_avail, n_cpu, num_workers = solv_num_workers()
    print(f"Number of available CPUs: {num_workers}")
    process_map(sdf2pyg_wrapper, mp_args, total=len(mp_args), chunksize=10, max_workers=num_workers, desc="proceesing FreeSolv-PHYSPROP")

def mask_mt_pyg_gen():
    intersection_csv = osp.join(RAW_DATA_ROOT, "FreeSolv_PHYSPROP", "intersection.csv")
    freesolv_pygs = glob(osp.join(TEMP_DATA_ROOT, "freesolv", "*.pyg"))
    physprop_pygs = glob(osp.join(TEMP_DATA_ROOT, "physprop", "*.pyg"))
    freesolv_csv = osp.join(RAW_DATA_ROOT, "FreeSolv_PHYSPROP", "freesolv_paper_fl.csv")
    physprop_csv = osp.join(RAW_DATA_ROOT, "FreeSolv_PHYSPROP", "logP_labels.csv")
    pyg_aux = osp.join(PROCESSED_DATA_ROOT, "freesolv_physprop_14k.pyg")

    intersection_df = pd.read_csv(intersection_csv, dtype={"sample_id_1": int, "sample_id_2": int})
    freesolv_id_intetersection: Set[int] = set(intersection_df["sample_id_1"].values.tolist())
    physprop_id_intersection: Set[int] = set(intersection_df["sample_id_2"].values.tolist())

    def _load_csv(csv):
        df = pd.read_csv(csv).rename({"FileHandle": "sample_id", "Kow": "activity"}, axis=1)
        return df.astype({"sample_id": int}).set_index("sample_id")
    freesolv_df = _load_csv(freesolv_csv)
    physprop_df = _load_csv(physprop_csv)

    data_list = []

    intersection_df = intersection_df.set_index("sample_id_1")
    for pyg_f in freesolv_pygs:
        try:
            pyg = torch.load(pyg_f)

            sample_id = int(osp.basename(pyg_f).split(".")[0])
            if sample_id in freesolv_id_intetersection:
                for key in PROPS:
                    setattr(pyg, key, torch.as_tensor([intersection_df.loc[sample_id][key]]))
                # fix an error in intersection.csv that CalcOct is missing a sign
                pyg.CalcOct = -pyg.CalcOct
                pyg.mask = torch.as_tensor([1, 1, 1, 1, 1, 1]).bool().view(1, -1)
            else:
                for key in PROPS:
                    setattr(pyg, key, torch.as_tensor([9999.]))
                setattr(pyg, "CalcSol", torch.as_tensor([freesolv_df.loc[sample_id]["activity"]]))
                pyg.mask = torch.as_tensor([0, 0, 0, 1, 0, 0]).bool().view(1, -1)
            pyg.freesolv_sample_id = sample_id

            data_list.append(pyg)
        except Exception as e:
            print(f"Error processing {pyg_f}: {e}")

    for pyg_f in physprop_pygs:
        try:
            pyg = torch.load(pyg_f)

            sample_id = int(osp.basename(pyg_f).split(".")[0])
            if sample_id not in physprop_id_intersection:
                for key in PROPS:
                    setattr(pyg, key, torch.as_tensor([9999.]))
                setattr(pyg, "watOct", torch.as_tensor([physprop_df.loc[sample_id]["activity"] * logP_to_watOct]))
                pyg.mask = torch.as_tensor([0, 0, 0, 0, 0, 1]).bool().view(1, -1)
                pyg.freesolv_sample_id = -1
                data_list.append(pyg)
        except Exception as e:
            print(f"Error processing {pyg_f}: {e}")

    res = torch_geometric.data.InMemoryDataset.collate(data_list)
    torch.save(res, pyg_aux)


if __name__ == "__main__":
    mp_proccess()
    mask_mt_pyg_gen()
