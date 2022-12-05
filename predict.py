import argparse
from collections import defaultdict
import torch
from torch_geometric.data import Data
from glob import glob
import numpy as np

from utils.gauss.read_gauss_log import Gauss16Log
from utils.DataPrepareUtils import my_pre_transform
from utils.trained_folder import TrainedFolder
from utils.utils_functions import get_device, kcal2ev
from utils.LossFn import logP_to_watOct

display_unit_mapper = {
    "E_gas": "eV", "E_water": "eV", "E_oct": "eV", "E_hydration": "kcal/mol", "LogP": ""
}


class SinglePredictor(TrainedFolder):
    def __init__(self, trained_model_folder, sdf_file):
        super().__init__(trained_model_folder, None)
        self.sdf_file = sdf_file

        self._data = None

    def display_prediction(self, keys):
        prediction = self.get_prediction()
        for key in keys:
            print(f"{key}: {prediction[key]} {display_unit_mapper[key]}")

    @property
    def data(self):
        """
        preprocessing into pytorch data
        """
        if self._data is None:
            sdf_info = Gauss16Log(log_path=None, log_sdf=self.sdf_file, supress_warning=True)
            this_data = Data(**sdf_info.get_basic_dict())
            this_data = my_pre_transform(this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                 cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                 bond_atom_sep=False, record_long_range=True)
            this_data.atom_mol_batch = torch.zeros_like(this_data.Z)
            this_data.BN_edge_index_correct = 0
            self._data = this_data
        return self._data

    def get_prediction(self):
        model_out = self.model(self.data.to(get_device()))
        mol_prop = model_out["mol_prop"].detach().cpu().numpy().reshape(-1)
        e_gas = mol_prop[0]
        e_water = mol_prop[1]
        e_oct = mol_prop[2]

        e_hydration = (e_water - e_gas) / kcal2ev
        e_wat_oct = (e_water - e_oct) / kcal2ev
        logp = e_wat_oct / logP_to_watOct

        out = {"E_gas": e_gas, "E_water": e_water, "E_oct": e_oct, "E_hydration": e_hydration, "LogP": logp}
        return out

class EnsPredictor:
    def __init__(self, trained_model_folders: str, sdf_file) -> None:
        self.trained_folders = glob(trained_model_folders)
        self.sdf_file = sdf_file

        self.single_predictors = [SinglePredictor(f, sdf_file) for f in self.trained_folders]

    def display_prediction(self, keys):
        prediction, std = self.get_prediction()
        for key in keys:
            print(f"{key}: {prediction[key]} +- {std[key]} {display_unit_mapper[key]}")
    
    def get_prediction(self):
        # avoid recalculation of the data
        for predictor in self.single_predictors[1:]:
            predictor._data = self.single_predictors[0].data

        predictions = [predictor.get_prediction() for predictor in self.single_predictors]
        if len(predictions) == 0:
            raise ValueError("Trained model not successfully downloaded. Please make sure 'bash bash_scripts/download_models_and_extract.bash' has finished successfully.")
        ens_prediciton = defaultdict(lambda: 0.)
        ens_std = defaultdict(lambda: [])

        for pred in predictions:
            for key in pred.keys():
                ens_prediciton[key] += pred[key]
                ens_std[key].append(pred[key])
        for key in ens_prediciton.keys():
            ens_prediciton[key] /= len(predictions)
        for key in ens_std.keys():
            ens_std[key] = np.std(np.asarray(ens_std[key])).item()
        return ens_prediciton, ens_std

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf")
    parser.add_argument("--model", help="cal_single | cal_ens5 | exp_ens5")
    args = parser.parse_args()

    if args.model == "cal_single":
        predictor = SinglePredictor("./data/models/exp_frag20sol_012_run_2022-04-20_143627", args.sdf)
        predictor.display_prediction(["E_gas", "E_water", "E_oct"])
    elif args.model == "cal_ens5":
        predictor = EnsPredictor("./data/models/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*", args.sdf)
        predictor.display_prediction(["E_gas", "E_water", "E_oct"])
    elif args.model == "exp_ens5":
        predictor = EnsPredictor("./data/models/exp_ultimate_freeSolv_13_RDrun_2022-05-20_100307__201005/exp_ultimate_freeSolv_13_active_ALL_2022-05-20_100309/exp_*_cycle_-1_*", args.sdf)
        predictor.display_prediction(["E_hydration", "LogP"])
    else:
        raise ValueError("Model must be one of the following: cal_single | cal_ens5 | exp_ens5")

if __name__ == "__main__":
    predict()
