import argparse
import glob
import json
import logging
import math
import os
import os.path as osp
import shutil
import sys
import time
from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from Networks.PhysDimeNet import PhysDimeNet
from Networks.UncertaintyLayers.swag import SWAG
from utils.DataPrepareUtils import my_pre_transform
from utils.DummyIMDataset import DummyIMDataset
from utils.LossFn import LossFn
from utils.Optimizers import EmaAmsGrad, MySGD
from utils.tags import tags
from utils.time_meta import print_function_runtime, record_data
from utils.utils_functions import add_parser_arguments, floating_type, preprocess_config, get_lr, collate_fn, \
    get_n_params, atom_mean_std, remove_handler, option_solver, fix_model_keys, get_device, process_state_dict, \
    validate_index, non_collapsing_folder


def remove_extra_keys(data_provider, logger=None, return_extra=False):
    # These are not needed during training
    remove_names = []
    extra = {}
    example_data = data_provider.data
    for key in example_data.keys:
        if not isinstance(getattr(example_data, key), torch.Tensor):
            remove_names.append(key)
    for name in remove_names:
        if return_extra:
            extra[name] = getattr(data_provider.data, name)
        delattr(data_provider.data, name)
    if logger is not None:
        logger.info(f"These keys are deleted during training: {remove_names}")

    if return_extra:
        return data_provider, extra
    return data_provider


def dataset_from_args(args, logger=None):
    default_kwargs = {'root': args["data_root"], 'pre_transform': my_pre_transform,
                      'record_long_range': True, 'type_3_body': 'B', 'cal_3body_term': True}
    data_provider_class, _kwargs = data_provider_solver(args["data_provider"], default_kwargs)
    _kwargs = _add_arg_from_config(_kwargs, args)
    data_provider: InMemoryDataset = data_provider_class(**_kwargs)

    data_provider = remove_extra_keys(data_provider, logger)

    return data_provider


def data_provider_solver(name_full, _kw_args):
    """

    :param name_full: Name should be in a format: ${name_base}[${key}=${value}], all key-value pairs will be feed into
    data_provider **kwargs
    :param _kw_args:
    :return: Data Provider Class and kwargs
    """
    additional_kwargs = option_solver(name_full)
    for key in additional_kwargs.keys():
        '''Converting string into corresponding data type'''
        if additional_kwargs[key] in ["True", "False"]:
            additional_kwargs[key] = (additional_kwargs[key] == "True")
        else:
            try:
                additional_kwargs[key] = float(additional_kwargs[key])
            except ValueError:
                pass
    name_base = name_full.split('[')[0]
    for key in additional_kwargs.keys():
        _kw_args[key] = additional_kwargs[key]

    assert name_base == "dummy", "All name_bases except 'dummy' are disabled"
    if name_base == 'qm9':
        from qm9InMemoryDataset import Qm9InMemoryDataset
        return Qm9InMemoryDataset, _kw_args
    elif name_base.split('_')[0] == 'frag20nHeavy':
        from Frag20IMDataset import Frag20IMDataset
        print("Deprecated dataset: {}".format(name_base))
        n_heavy_atom = int(name_base[7:])
        _kw_args['n_heavy_atom'] = n_heavy_atom
        return Frag20IMDataset, _kw_args
    elif name_base[:9] == 'frag9to20':
        from Frag9to20MixIMDataset import Frag9to20MixIMDataset, uniform_split, small_split, large_split
        _kw_args['training_option'] = 'train'
        split = name_base[10:]
        if split == 'uniform':
            _kw_args['split_settings'] = uniform_split
        elif split == 'small':
            _kw_args['split_settings'] = small_split
        elif split == 'large':
            _kw_args['split_settings'] = large_split
        elif split == 'all':
            _kw_args['split_settings'] = uniform_split
            _kw_args['all_data'] = True
        elif split == 'jianing':
            _kw_args['split_settings'] = uniform_split
            _kw_args['jianing_split'] = True
            _kw_args["all_data"] = False
        else:
            raise ValueError('not recognized argument: {}'.format(split))
        return Frag9to20MixIMDataset, _kw_args
    elif name_base in ['frag20_eMol9_combine', 'frag20_eMol9_combine_MMFF']:
        from PhysDimeIMDataset import PhysDimeIMDataset
        from CombinedIMDataset import CombinedIMDataset
        geometry = "MMFF" if name_base == 'frag20_eMol9_combine_MMFF' else "QM"
        frag20dataset, tmp_args = data_provider_solver('frag9to20_all', _kw_args)
        frag20dataset = frag20dataset(**tmp_args, geometry=geometry)
        len_frag20 = len(frag20dataset)
        val_index = frag20dataset.val_index
        train_index = frag20dataset.train_index
        _kw_args['dataset_name'] = 'frag20_eMol9_combined_{}.pt'.format(geometry)
        _kw_args['val_index'] = val_index
        e_mol9_dataset = PhysDimeIMDataset(root=tmp_args['root'], processed_prefix='eMol9_{}'.format(geometry),
                                           pre_transform=my_pre_transform,
                                           record_long_range=tmp_args['record_long_range'],
                                           infile_dic={
                                               'PhysNet': 'eMol9_PhysNet_{}.npz'.format(geometry),
                                               'SDF': 'eMol9_{}.pt'.format(geometry)})
        len_e9 = len(e_mol9_dataset)
        _kw_args['train_index'] = torch.cat([train_index, torch.arange(len_frag20, len_e9 + len_frag20)])
        _kw_args['dataset_list'] = [frag20dataset, e_mol9_dataset]
        return CombinedIMDataset, _kw_args
    elif name_base == "dummy":
        assert "dataset_name", "split" in additional_kwargs
        _kw_args.update(additional_kwargs)
        return DummyIMDataset, _kw_args
    else:
        raise ValueError('Unrecognized dataset name: {} !'.format(name_base))


def train_step(model, _optimizer, data_batch, loss_fn, max_norm, scheduler, config_dict):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.autograd.set_detect_anomaly(False):
        model.train()
        _optimizer.zero_grad()

        model_out = model(data_batch)

        if config_dict["time_debug"]:
            t0 = record_data('forward', t0, True)

        loss = loss_fn(model_out, data_batch, True) + model_out["nh_loss"]

        if config_dict["time_debug"]:
            t0 = record_data('loss_cal', t0, True)

        loss.backward()

        if config_dict["time_debug"]:
            t0 = record_data('backward', t0, True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    _optimizer.step()
    if config_dict["scheduler_base"] in tags.step_per_step:
        scheduler.step()

    if config_dict["time_debug"]:
        t0 = record_data('step', t0, True)
    # print_function_runtime()

    result_loss = loss.data[0]

    return result_loss


def val_step_new(model, _data_loader, loss_fn, diff=False, lightweight=True, config_dict=None):
    model.eval()
    valid_size = 0
    # This is used in flexible training where only part of the properties are present in molecules
    flex_size = {}
    loss = 0.
    detail = None
    with torch.set_grad_enabled(False):
        for val_data in _data_loader:
            val_data = val_data.to(get_device())
            if config_dict is not None and config_dict["mem_debug"]:
                # TODO: why is it still printing??
                pass
                # print(torch.cuda.memory_summary())
                # print("---")
            model_out = model(val_data)
            aggr_loss, loss_detail = loss_fn(model_out, val_data, False, loss_detail=True, diff_detail=diff)
            # n_units is the batch size when predicting mol props but number of atoms when predicting atom props.
            n_units = loss_detail["n_units"]
            loss += aggr_loss.item() * n_units
            if detail is None:
                # -----init------ #
                detail = copy(loss_detail)

                for key in loss_detail.keys():
                    if tags.val_avg(key):
                        detail[key] = 0.
                    elif tags.val_concat(key):
                        # :param lightweight: to make the final file small, otherwise it grows into several GBs
                        if lightweight and key == "atom_embedding":
                            del detail[key]
                        else:
                            detail[key] = []
                    else:
                        # we do not want temp information being stored in the final csv file
                        del detail[key]

            for key in detail:
                if tags.val_avg(key):
                    if "flex_units" in loss_detail.keys():
                        prop_name = key.split("_")[-1]
                        detail[key] += loss_detail[key] * loss_detail["flex_units"][prop_name]
                        if key not in flex_size.keys():
                            flex_size[key] = 0
                        flex_size[key] += loss_detail["flex_units"][prop_name]
                    else:
                        detail[key] += loss_detail[key] * n_units
                elif tags.val_concat(key) and key not in ["ATOM_MOL_BATCH"]:
                    detail[key].append(loss_detail[key])
                elif key == "ATOM_MOL_BATCH":
                    detail[key].append(loss_detail[key] + valid_size)
            valid_size += n_units
    detail["n_units"] = valid_size
    for key in flex_size:
        detail[f"n_units_{key}"] = flex_size[key]

    loss /= valid_size
    # Stacking if/else like hell
    for key in list(detail.keys()):
        if tags.val_avg(key):
            if key in flex_size.keys():
                if flex_size[key] == 0:
                    detail[key] = None
                else:
                    detail[key] /= flex_size[key]
            else:
                detail[key] /= valid_size
        elif tags.val_concat(key):
            detail[key] = torch.cat(detail[key], dim=0)

        if key.startswith("MSE_"):
            if detail[key] is not None:
                detail[f"RMSE_{key.split('MSE_')[-1]}"] = math.sqrt(detail[key])
            else:
                detail[f"RMSE_{key.split('MSE_')[-1]}"] = None
    detail["loss"] = loss
    return detail


def train(config_dict=None, data_provider=None, explicit_split=None, ignore_valid=False, use_tqdm=False):
    # ------------------- variable set up ---------------------- #
    config_dict = preprocess_config(config_dict)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if data_provider is None:
        data_provider = dataset_from_args(config_dict, logger)
    logger.info("used dataset: {}".format(data_provider.processed_file_names))

    # ----------------- set up run directory -------------------- #
    folder_prefix = config_dict["folder_prefix"]
    tmp = osp.basename(folder_prefix)
    run_directory = non_collapsing_folder(folder_prefix)
    shutil.copyfile(config_dict["config_name"], osp.join(run_directory, f"config-{tmp}.txt"))
    with open(osp.join(run_directory, "config_runtime.json"), "w") as out:
        json.dump(config_dict, out, skipkeys=True, indent=4, default=lambda x: None)

        # --------------------- Logger setup ---------------------------- #
        # first we want to remove previous logger step up by other programs
        # we sometimes encounter issues and the logger file doesn't show up
        log_tmp = logging.getLogger()
        remove_handler(log_tmp)

        logging.basicConfig(filename=osp.join(run_directory, config_dict["log_file_name"]),
                            format='%(asctime)s %(message)s', filemode='w')

        # -------------------- Meta data file set up -------------------- #
        meta_data_name = os.path.join(run_directory, 'meta.txt')
        print("run dir: {}".format(run_directory))

    # -------------- Index file and remove specific atoms ------------ #
    train_index, val_index, test_index = data_provider.train_index, data_provider.val_index, data_provider.test_index

    if config_dict["debug_mode"]:
        train_index = train_index[:1000]
        logger.warning("***********DEBUG MODE ON, Result not trustworthy***************")
    train_size, val_size, test_size = validate_index(train_index, val_index, test_index)
    logger.info(f"train size: {train_size}")
    logger.info(f"validation size: {val_size}")
    logger.info(f"test size: {test_size}")

    n_cpu_avail = len(os.sched_getaffinity(0))
    n_cpu = os.cpu_count()
    # screw it, the num_workers is really problematic, causing deadlocks
    num_workers = 0
    logger.info(f"Number of total CPU: {n_cpu}")
    logger.info(f"Number of available CPU: {n_cpu_avail}")

    # dataloader
    loader_kw_args = {"shuffle": True, "batch_size": config_dict["batch_size"]}
    if config_dict["over_sample"]:
        has_solv_mask = data_provider.data.mask[torch.as_tensor(train_index), 3]
        n_total = has_solv_mask.shape[0]
        n_has_wat_solv = has_solv_mask.sum().item()
        n_no_wat_solv = n_total - n_has_wat_solv
        logger.info(f"Oversampling: {n_total} molecules are in the training set")
        logger.info(f"Oversampling: {n_has_wat_solv} molecules has water solv")
        logger.info(f"Oversampling: {n_no_wat_solv} molecules do not have water solv")
        weights = torch.zeros_like(has_solv_mask).float()
        weights[has_solv_mask] = n_no_wat_solv
        weights[~has_solv_mask] = n_has_wat_solv
        sampler = WeightedRandomSampler(weights=weights, num_samples=n_total)
        loader_kw_args["sampler"] = sampler
        loader_kw_args["shuffle"] = False
    train_data_loader = torch.utils.data.DataLoader(
        data_provider[torch.as_tensor(train_index)], collate_fn=collate_fn, pin_memory=True, num_workers=num_workers, **loader_kw_args)
    val_data_loader = torch.utils.data.DataLoader(
        data_provider[torch.as_tensor(val_index)], batch_size=config_dict["valid_batch_size"], collate_fn=collate_fn,
        pin_memory=True, shuffle=False, num_workers=num_workers)

    w_e, w_f, w_q, w_p = 1., config_dict["force_weight"], config_dict["charge_weight"], config_dict["dipole_weight"]
    loss_fn = LossFn(w_e=w_e, w_f=w_f, w_q=w_q, w_p=w_p, action=config_dict["action"], auto_sol=config_dict["auto_sol"],
                     target_names=config_dict["target_names"], config_dict=config_dict)

    # ------------------- Setting up model and optimizer ------------------ #
    # Normalization of PhysNet atom-wise prediction
    assert config_dict["action"] in ["names", "names_and_QD"], config_dict["action"]
    mean_atom = []
    std_atom = []
    for name in config_dict["target_names"]:
        this_mean, this_std = atom_mean_std(getattr(data_provider.data, name), data_provider.data.N, train_index)
        mean_atom.append(this_mean)
        std_atom.append(this_std)
    if config_dict["action"] == "names_and_QD":
        # the last dimension is for predicting atomic charge
        mean_atom.append(0.)
        std_atom.append(1.)
    mean_atom = torch.as_tensor(mean_atom)
    std_atom = torch.as_tensor(std_atom)

    E_atomic_scale = std_atom
    E_atomic_shift = mean_atom

    config_dict['energy_shift'] = E_atomic_shift
    config_dict['energy_scale'] = E_atomic_scale


    net = PhysDimeNet(**config_dict)
    net = net.to(get_device())
    net = net.type(floating_type)

    # model freeze options (transfer learning)
    if config_dict["freeze_option"] == 'prev':
        net.freeze_prev_layers(freeze_extra=False)
    elif config_dict["freeze_option"] == 'prev_extra':
        net.freeze_prev_layers(freeze_extra=True)
    elif config_dict["freeze_option"] == 'none':
        pass
    else:
        raise ValueError('Invalid freeze option: {}'.format(config_dict["freeze_option"]))

    # ---------------------- restore pretrained model ------------------------ #
    if config_dict["use_trained_model"]:
        model_chk = config_dict["use_trained_model"]
        trained_model_dir = glob.glob(model_chk)
        assert len(trained_model_dir) == 1, f"Zero or multiple trained folder: {trained_model_dir}"
        trained_model_dir = trained_model_dir[0]
        config_dict["use_trained_model"] = trained_model_dir
        logger.info('using trained model: {}'.format(trained_model_dir))
        
        train_model_path = osp.join(trained_model_dir, 'training_model.pt')
        if not osp.exists(train_model_path):
            train_model_path = osp.join(trained_model_dir, 'best_model.pt')

        best_model_path = osp.join(trained_model_dir, 'best_model.pt')
        for _net, _model_path in zip([net], [train_model_path]):
            state_dict = torch.load(_model_path, map_location=get_device())
            state_dict = fix_model_keys(state_dict)
            state_dict = process_state_dict(state_dict, config_dict, logger)

            incompatible_keys = _net.load_state_dict(state_dict=state_dict, strict=False)

            logger.info(f"---------vvvvv incompatible keys in {_model_path} vvvvv---------")
            logger.info(str(incompatible_keys))
        shadow_dict = torch.load(best_model_path, map_location=get_device())
        shadow_dict = process_state_dict(fix_model_keys(shadow_dict), config_dict, logger)
    else:
        shadow_dict = None

    # optimizers
    ema_decay = config_dict["ema_decay"]
    assert config_dict["optimizer"].split('_')[0] == 'emaAms', config_dict["optimizer"]
    assert float(config_dict["optimizer"].split('_')[1]) == ema_decay
    optimizer = EmaAmsGrad(net, lr=config_dict["learning_rate"], ema=float(config_dict["optimizer"].split('_')[1]),
                            shadow_dict=shadow_dict)

    # schedulers
    scheduler_kw_args = option_solver(config_dict["scheduler"], type_conversion=True)
    scheduler_base = config_dict["scheduler"].split("[")[0]
    config_dict["scheduler_base"] = scheduler_base
    if scheduler_base == "StepLR":
        if "decay_epochs" in scheduler_kw_args.keys():
            step_per_epoch = 1. * train_size / config_dict["batch_size"]
            decay_steps = math.ceil(scheduler_kw_args["decay_epochs"] * step_per_epoch)
        else:
            decay_steps = config_dict["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_steps, gamma=0.1)
    elif scheduler_base == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kw_args)
        assert config_dict["warm_up_steps"] == 0
    else:
        raise ValueError('Unrecognized scheduler: {}'.format(config_dict["scheduler"]))

    if config_dict["use_trained_model"] and (not config_dict["reset_optimizer"]):
        if os.path.exists(os.path.join(trained_model_dir, "best_model_optimizer.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_optimizer.pt"),
                                                 map_location=get_device()))
        if os.path.exists(os.path.join(trained_model_dir, "best_model_scheduler.pt")):
            scheduler.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_scheduler.pt"),
                                                 map_location=get_device()))

    if config_dict["use_trained_model"]:
        # protect LR been reduced
        step_per_epoch = 1. * train_size / config_dict["batch_size"]
        ema_avg = 1 / (1 - ema_decay)
        ft_protection = math.ceil(ema_avg / step_per_epoch)
        ft_protection = min(30, ft_protection)
    else:
        ft_protection = 0
    logger.info(f"Fine tune protection: {ft_protection} epochs. ")

    # --------------------- Printing meta data ---------------------- #
    if get_device().type == 'cuda':
        logger.info('Hello from device : ' + torch.cuda.get_device_name(get_device()))
        logger.info("Cuda mem allocated: {:.2f} MB".format(torch.cuda.memory_allocated(get_device()) * 1e-6))

    with open(meta_data_name, 'w+') as f:
        n_parm, model_structure = get_n_params(net, None, False)
        logger.info('model params: {}'.format(n_parm))
        f.write('*' * 20 + '\n')
        f.write("all params\n")
        f.write(model_structure)
        f.write('*' * 20 + '\n')
        n_parm, model_structure = get_n_params(net, None, True)
        logger.info('trainable params: {}'.format(n_parm))
        f.write('*' * 20 + '\n')
        f.write("trainable params\n")
        f.write(model_structure)
        f.write('*' * 20 + '\n')
        f.write('train data index:{} ...\n'.format(train_index[:100]))
        f.write('val data index:{} ...\n'.format(val_index[:100]))
        # f.write('test data index:{} ...\n'.format(test_index[:100]))
        for _key in config_dict.keys():
            f.write("{} = {}\n".format(_key, config_dict[_key]))

    # ---------------------- Training ----------------------- #
    logger.info('start training...')

    shadow_net = optimizer.shadow_model
    val_res = val_step_new(shadow_net, val_data_loader, loss_fn)

    # csv files recording all loss info
    csv_f = osp.join(run_directory, "loss_data.csv")
    if osp.exists(csv_f):
        loss_df = pd.read_csv(csv_f)
    else:
        loss_df = pd.DataFrame()
        this_df_dict = OrderedDict(
            {"epoch": 0, "train_loss": -1, "valid_loss": val_res["loss"], "delta_time": time.time() - t0})
        for key in val_res.keys():
            if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                this_df_dict[key] = val_res[key]
                this_df_dict.move_to_end(key)
        loss_df = loss_df.append(this_df_dict, ignore_index=True)
    loss_df.to_csv(csv_f, index=False)

    # use np.inf instead of val_res["loss"] for proper transfer learning behaviour
    best_loss = np.inf

    last_epoch = pd.read_csv(osp.join(run_directory, "loss_data.csv"), header="infer").iloc[-1]["epoch"]
    last_epoch = int(last_epoch.item())
    logger.info('Init lr: {}'.format(get_lr(optimizer)))

    early_stop_count = 0
    step = 0

    logger.info("Setup complete, training starts...")

    for epoch in range(last_epoch, last_epoch + config_dict["num_epochs"]):

        # Early stop when learning rate is too low
        this_lr = get_lr(optimizer)
        if step > config_dict["warm_up_steps"] and config_dict["stop_low_lr"] and this_lr < 3*getattr(scheduler, "eps", 1e-9):
            logger.info('early stop because of low LR at epoch {}.'.format(epoch))
            break

        loader = enumerate(train_data_loader)
        if use_tqdm:
            loader = tqdm(loader, "training")

        train_loss = 0.
        for batch_num, data in loader:
            data = data.to(get_device())
            this_size = data.N.shape[0]

            train_loss += train_step(net, _optimizer=optimizer, data_batch=data, loss_fn=loss_fn,
                                     max_norm=config_dict["max_norm"], scheduler=scheduler,
                                     config_dict=config_dict) * this_size / train_size
            step += 1

        # ---------------------- Post training steps: validation, save model, print meta ---------------- #
        logger.info('epoch {} ended, learning rate: {} '.format(epoch, this_lr))
        shadow_net = optimizer.shadow_model
        val_res = val_step_new(shadow_net, val_data_loader, loss_fn)
        if config_dict["scheduler_base"] in tags.step_per_epoch and (epoch - last_epoch) >= ft_protection:
            scheduler.step(metrics=val_res["loss"])

        this_df_dict = {"epoch": epoch, "train_loss": train_loss.cpu().item(), "valid_loss": val_res["loss"],
                        "delta_time": time.time() - t0}
        for key in val_res.keys():
            if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                this_df_dict[key] = val_res[key]

        loss_df = loss_df.append(this_df_dict, ignore_index=True)
        loss_df.to_csv(csv_f, index=False)

        if ignore_valid or (val_res['loss'] < best_loss):
            early_stop_count = 0
            best_loss = val_res['loss']
            torch.save(shadow_net.state_dict(), osp.join(run_directory, 'best_model.pt'))
            torch.save(net.state_dict(), osp.join(run_directory, 'training_model.pt'))
            torch.save(optimizer.state_dict(), osp.join(run_directory, 'best_model_optimizer.pt'))
            torch.save(scheduler.state_dict(), osp.join(run_directory, "best_model_scheduler.pt"))
        else:
            early_stop_count += 1
            if early_stop_count == config_dict["early_stop"]:
                logger.info('early stop at epoch {}.'.format(epoch))
                break

    remove_handler(logger)
    meta = {"run_directory": run_directory}
    return meta


def _add_arg_from_config(_kwargs, config_args):
    for attr_name in ['edge_version', 'cutoff', 'boundary_factor']:
        _kwargs[attr_name] = config_args[attr_name]
    return _kwargs


def flex_parse(add_extra_args=None):
    # set up parser and arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)
    if add_extra_args is not None:
        parser = add_extra_args(parser)

    # parse config file
    if len(sys.argv) == 1:
        config_name = 'config.txt'
        if os.path.isfile(config_name):
            args, unknown = parser.parse_known_args(["@" + config_name])
        else:
            raise Exception(f"Cannot find {config_name}")
    else:
        args = parser.parse_args()
        config_name = args.config_name
        args, unknown = parser.parse_known_args(["@" + config_name])
    args.config_name = config_name
    args = vars(args)

    args["local_rank"] = int(os.environ.get("LOCAL_RANK") if os.environ.get("LOCAL_RANK") is not None else 0)
    args["is_dist"] = (os.environ.get("RANK") is not None)

    return args


def main():
    args = flex_parse()

    train(args)


if __name__ == "__main__":
    main()
