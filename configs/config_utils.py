"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from common import PATHS


class DEFAULT_CONFIGS_PATHS:
    SCAN_FLOWERS = os.path.join(PATHS.PROJECT_ROOT, 'configs/scan_flowers.yml')
    SIMCLR_FLOWERS = os.path.join(PATHS.PROJECT_ROOT, 'configs/simclr_flowers.yml')


def save_cfg_yaml(cfg, file_path: str):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)


def prepare_config_paths(cfg, root_dir: str = PATHS.OUT_ROOT, exp_name: str = None):
    # Config for environment path
    assert os.path.isdir(root_dir)

    if exp_name is not None:
        root_dir = os.path.join(root_dir, exp_name)

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(pretext_dir, exist_ok=True)

    cfg['cfg_path'] = os.path.join(base_dir, 'config.yaml')
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')
    cfg['fist_batch_val_path'] = os.path.join(pretext_dir, 'fist_batch_val_path.npy')

    cfg['knn_val'] = os.path.join(pretext_dir, 'knn_val.npy')
    cfg['knn_train'] = os.path.join(pretext_dir, 'knn_train.npy')
    cfg['wiegths_knn_val'] = os.path.join(pretext_dir, 'wiegths_knn_val.npy')

    cfg['class_to_idx_tr'] = os.path.join(pretext_dir, 'class_to_idx_tr.npy')
    cfg['class_to_idx_val'] = os.path.join(pretext_dir, 'class_to_idx_val.npy')

    cfg['features_val_path'] = os.path.join(pretext_dir, 'features_val_path.npy')
    cfg['features_tr_path'] = os.path.join(pretext_dir, 'features_tr_path.npy')

    cfg['targets_val_path'] = os.path.join(pretext_dir, 'targets_val_path.npy')
    cfg['targets_tr_path'] = os.path.join(pretext_dir, 'targets_tr_path.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'scan')
        selflabel_dir = os.path.join(base_dir, 'selflabel') 
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(scan_dir, exist_ok=True)
        os.makedirs(selflabel_dir, exist_ok=True)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')

    return cfg


def load_config_params(config_yaml_file_path: str) -> EasyDict:
    with open(config_yaml_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()
    # Copy
    for k, v in config.items():
        cfg[k] = v
    return cfg


def get_config_params(config_file_path: str = None, scenario: str = 'scan', data_type: str = 'flowers') -> EasyDict:
    if config_file_path is None:
        assert data_type == 'flowers'
        if scenario == 'scan':
            config_file_path = DEFAULT_CONFIGS_PATHS.SCAN_FLOWERS
        elif scenario == 'simclr':
            config_file_path = DEFAULT_CONFIGS_PATHS.SIMCLR_FLOWERS
        else:
            raise NameError(f'Not valid scenario name: {scenario}')

    cfg_params = load_config_params(config_file_path)
    return cfg_params
