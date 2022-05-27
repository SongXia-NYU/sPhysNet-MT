import argparse
import glob

import torch

from utils.tester import Tester


def test_folder(folder_name, n_forward, x_forward, use_exist=False, check_active=False, ignore_train=True,
                lightweight=True, config_folder=None, no_runtime_split=False):

    tester = Tester(folder_name, n_forward, x_forward, use_exist, check_active, ignore_train, lightweight,
                    config_folder, no_runtime_split)
    tester.run_test()


def test_all():
    parser = argparse.ArgumentParser()
    # parser = add_parser_arguments(parser)
    parser.add_argument('--folder_names', default='exp_freeSolv14_run_2022-04-13_113008', type=str)
    parser.add_argument('--config_folders', default=None, type=str)
    parser.add_argument('--x_forward', default=1, type=int)
    parser.add_argument('--n_forward', default=25, type=int)
    parser.add_argument('--use_exist', action="store_true")
    parser.add_argument('--include_train', action="store_false")
    parser.add_argument('--heavyweight', action="store_false")
    parser.add_argument("--no_runtime_split", action="store_true")
    _args = parser.parse_args()

    run_dirs = glob.glob(_args.folder_names)

    if _args.config_folders is not None:
        config_folders = glob.glob(_args.config_folders)
        assert len(config_folders) == 1
        config_folder = config_folders[0]
    else:
        config_folder = None

    for name in run_dirs:
        print('testing folder: {}'.format(name))
        test_folder(name, _args.n_forward, _args.x_forward, _args.use_exist, ignore_train=_args.include_train,
                    lightweight=_args.heavyweight, config_folder=config_folder, no_runtime_split=_args.no_runtime_split)


def cal_loss(pred, e_target, d_target, q_target, mae_fn, mse_fn):
    # TODO: I do not remember what it is used for, but I do not dare to delete it
    result = {
        'loss': None,
        'emae': mae_fn(pred['E_pred'], e_target),
        'ermse': torch.sqrt(mse_fn(pred['E_pred'], e_target)),
        'pmae': mae_fn(pred['D_pred'], d_target),
        'prmse': torch.sqrt(mse_fn(pred['D_pred'], d_target)),
        'qmae': mae_fn(pred['Q_pred'], q_target),
        'qrmse': torch.sqrt(mse_fn(pred['Q_pred'], q_target)),
    }
    return result


if __name__ == "__main__":
    test_all()
