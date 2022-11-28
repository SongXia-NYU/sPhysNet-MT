import argparse
import torch
from torch_geometric.data import Data

from utils.gauss.read_gauss_log import Gauss16Log
from utils.DataPrepareUtils import my_pre_transform
from utils.trained_folder import TrainedFolder


class SinglePredictor(TrainedFolder):
    def __init__(self, trained_model_folder, sdf_file):
        super().__init__(trained_model_folder, None)
        self.sdf_file = sdf_file

        self._data = None

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

    def predict(self):
        model_out = self.model(self.data)
        print(model_out)

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf")
    parser.add_argument("--model", help="cal_single | cal_ens5 | exp_ens5")
    args = parser.parse_args()

    predictor = SinglePredictor("./data/models/exp_frag20sol_012_run_2022-04-20_143627", args.sdf)
    predictor.predict()

if __name__ == "__main__":
    predict()
