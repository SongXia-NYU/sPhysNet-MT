from collections import defaultdict
from glob import glob
import os.path as osp
import torch

from utils.concat_pyg import concat_pyg
from utils.frag20_sol_single import PROCESSED_DATA_ROOT, TEMP_DATA_ROOT, RAW_DATA_ROOT


FRAG_TEMPLATE = "frag20-sol-{}.pyg"
CSD20_TEMPLATE = "csd20-sol-{}.pyg"
CONF20_TEMPLATE = "conf20-sol-{}.pyg"

RETAIN_KEYS = ["CalcSol", "CalcOct", "calcLogP", "watOct", "gasEnergy", "watEnergy", "octEnergy", "R", "Z", "N",
               "BN_edge_index", "num_BN_edge", "sample_id", "dataset_name"]
RETAIN_KEYS = set(RETAIN_KEYS)


def _process_dataset(data, slices, name):
    length = len(data.N)

    def _rename(b4, after):
        setattr(data, after, getattr(data, b4))
        slices[after] = slices[b4]

    if name == "conf20":
        # _rename("water_gas(kcal/mol)", "CalcSol")
        # _rename("1-octanol_gas(kcal/mol)", "CalcOct")
        # _rename("water_1-octanol(kcal/mol)", "watOct")
        # data.split = ["train"]*length
        # slices["split"] = slices["N"]
        pass
    else:
        _rename("FileHandle", "sample_id")

    setattr(data, "dataset_name", [name] * length)
    slices["dataset_name"] = slices["sample_id"]

    for key in data.keys:
        if key not in RETAIN_KEYS:
            delattr(data, key)
            del slices[key]


def prepare_frag20_sol_678k():
    for geometry in ["mmff"]:
        frag20_data, frag20_slices = torch.load(osp.join(PROCESSED_DATA_ROOT, FRAG_TEMPLATE.format(geometry)))
        _process_dataset(frag20_data, frag20_slices, "frag20")
        out_data = frag20_data
        out_slices = frag20_slices

        csd20_data, csd20_slices = torch.load(osp.join(PROCESSED_DATA_ROOT, CSD20_TEMPLATE.format(geometry)))
        _process_dataset(csd20_data, csd20_slices, "csd20")

        conf20_data, conf20_slices = torch.load(osp.join(PROCESSED_DATA_ROOT, CONF20_TEMPLATE.format(geometry)))
        _process_dataset(conf20_data, conf20_slices, "conf20")

        for key in RETAIN_KEYS:
            print(f"processing key: {key}")
            is_list = (key in ["sample_id", "dataset_name"])
            this_frag20_val = getattr(frag20_data, key)
            this_csd20_val = getattr(csd20_data, key)
            this_conf20_val = getattr(conf20_data, key)
            if key == "sample_id":
                this_conf20_val = this_conf20_val.numpy().tolist()
            if key.endswith("_index"):
                correction_csd20 = this_frag20_val.shape[-1]
                correction_conf20 = this_csd20_val.shape[-1] + correction_csd20
                cat_dim = -1
            else:
                if is_list:
                    correction_csd20 = len(this_frag20_val)
                    correction_conf20 = len(this_csd20_val) + correction_csd20
                else:
                    correction_csd20 = this_frag20_val.shape[0]
                    correction_conf20 = this_csd20_val.shape[0] + correction_csd20
                cat_dim = 0
            this_frag20_slice = frag20_slices[key]
            this_csd20_slice = csd20_slices[key] + correction_csd20
            this_conf20_slice = conf20_slices[key] + correction_conf20
            slices_outs = [this_frag20_slice, this_csd20_slice, this_conf20_slice]
            if is_list:
                this_frag20_val.extend(this_csd20_val)
                this_frag20_val.extend(this_conf20_val)
                setattr(out_data, key, this_frag20_val)
            else:
                val_outs = [this_frag20_val, this_csd20_val, this_conf20_val]
                setattr(out_data, key, torch.cat(val_outs, dim=cat_dim))
            out_slices[key] = torch.cat(slices_outs, dim=0)

        n_total = len(out_data.N)

        torch.save((out_data, out_slices),
                   osp.join(PROCESSED_DATA_ROOT, f"frag20-sol-678k-{geometry}.pyg"))

        explicit_split = torch.load(osp.join(RAW_DATA_ROOT, "frag20-solv-678k-split.pth"))
        split = defaultdict(lambda: [])
        for i, sample_id in enumerate(out_data.sample_id):
            for split_name in explicit_split.keys():
                if sample_id in explicit_split[split_name]:
                    split[split_name].append(i)
                    continue
                
        torch.save(dict(split), osp.join(PROCESSED_DATA_ROOT, f"frag20-sol-678k-{geometry}-split.pyg"))
        for key in split.keys():
            print(f"{key} size: {len(split[key])}")
        print("Frag20-solv-678k preprocessing complete!")


def concat_frag20():
    for geometry in ["mmff"]:
        pygs = glob(f"{TEMP_DATA_ROOT}/frag20/{geometry}/*.pyg")

        concat_pyg(pygs, osp.join(PROCESSED_DATA_ROOT, f"frag20-sol-{geometry}.pyg"), use_mp=False,
         del_keys=["QM_SMILES", "QM_InChI", "ID", "SourceFile", "L_edge_index", "num_L_edge"])

def concat_csd20():
    for geometry in ["mmff"]:
        pygs = glob(f"{TEMP_DATA_ROOT}/csd20/{geometry}/*.pyg")
        concat_pyg(pygs, osp.join(PROCESSED_DATA_ROOT, f"csd20-sol-{geometry}.pyg"), use_mp=False,
         del_keys=["QM_SMILES", "QM_InChI", "ID", "SourceFile", "L_edge_index", "num_L_edge"])

def concat_conf20():
    for geometry in ["mmff"]:
        pygs = glob(f"{TEMP_DATA_ROOT}/conf20/{geometry}/*.pyg")
        concat_pyg(pygs, osp.join(PROCESSED_DATA_ROOT, f"conf20-sol-{geometry}.pyg"), use_mp=False,
         del_keys=["water_smiles", "gas_smiles", "1-octanol_smiles", "L_edge_index", "num_L_edge"])

if __name__ == "__main__":
    concat_frag20()
    concat_csd20()
    concat_conf20()

    prepare_frag20_sol_678k()
