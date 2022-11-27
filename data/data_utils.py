import os
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset

from data.custom_dataset import NeighborsDataset, AugmentedDataset, AugmentedPyramidDataset
# from data.cifar import CIFAR10, STL10, ImageNet, ImageNetSubset
from common import PATHS
from data.flowers_ds import FlowersDatasets, FlowersDatasetsPyramidImgs
from data.utils import collate_custom


def get_train_dataset(config_params,
                      transform,
                      root_ds_path: str = PATHS.FLOWERS_DS_KAGGLE,
                      to_augmented_dataset: bool = False,
                      to_neighbors_dataset: bool = False,
                      split: Optional[str] = None,
                      pretext_pretrained_path: Optional[str] = None) -> Dataset:
    # Base dataset
    if config_params['train_db_name'] == 'flowers-data':
        to_tensor = True  # (p['setup'] != 'scan')
        no_labels = (split == 'train+unlabeled')
        if config_params['use_patches']:
            dataset = FlowersDatasetsPyramidImgs(root=root_ds_path,
                                                 splitted=False,
                                                 split='train',
                                                 transform=transform,
                                                 patch_size=config_params['patch_size'],
                                                 img_size=config_params['img_size'],
                                                 step_scale=config_params['patch_overlap_scale'],
                                                 to_tensor=to_tensor,
                                                 no_labels=no_labels)
        else:
            dataset = FlowersDatasets(root=root_ds_path,
                                      split='train',
                                      transform=transform,
                                      img_size=config_params['img_size'],
                                      splitted=False,
                                      to_tensor=to_tensor,
                                      no_labels=no_labels)
            # dataset = torch.utils.data.ConcatDataset([dataset_keggel, dataset_origin])

    else:
        raise ValueError('Invalid train dataset {}'.format(config_params['train_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset:  # Dataset returns an image and an augmentation of that image.
        if config_params['train_db_name'] == 'flowers-data' and config_params['use_patches']:
            dataset = AugmentedPyramidDataset(dataset, aug_kwargs=config_params['augmentation_kwargs'])
        else:
            dataset = AugmentedDataset(dataset, aug_kwargs=config_params['augmentation_kwargs'])

    if to_neighbors_dataset:  # Dataset returns an image and one of its nearest neighbors.
        # indices = np.load(p['topk_neighbors_train_path'])
        knn_path = config_params['knn_train']
        if pretext_pretrained_path is not None:
            knn_path = os.path.join(pretext_pretrained_path, os.path.basename(config_params['knn_train']))
        assert os.path.isfile(knn_path), f'not exist file {knn_path}'
        indices = np.load(knn_path)
        dataset = NeighborsDataset(dataset, indices, config_params['num_neighbors'],
                                   aug_kwargs=config_params['augmentation_kwargs'],
                                   all_neighbors=config_params['all_neighbors'])

    return dataset


def get_val_dataset(config_params: dict, transform=None, to_neighbors_dataset=False,
                    pretext_pretrained_path: str = None,
                    root_ds_path: str = PATHS.FLOWERS_DS_UNLABELED):
    # Base dataset
    if config_params['val_db_name'] == 'flowers-data':
        to_tensor = True  # (p['setup'] != 'scan')
        if config_params['use_patches']:
            dataset = FlowersDatasetsPyramidImgs(split='test',
                                                 root=root_ds_path,
                                                 transform=transform,
                                                 patch_size=config_params['patch_size'],
                                                 img_size=config_params['img_size'],
                                                 step_scale=config_params['patch_overlap_scale'],
                                                 to_tensor=to_tensor,
                                                 no_labels=False)
        else:
            dataset = FlowersDatasets(split='test',
                                      root=root_ds_path,
                                      transform=transform,
                                      img_size=config_params['img_size'],
                                      to_tensor=to_tensor,
                                      no_labels=False)

    else:
        raise ValueError('Invalid validation dataset {}'.format(config_params['val_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_neighbors_dataset:  # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        # indices = np.load(p['topk_neighbors_val_path'])
        knn_path = config_params['knn_val']
        if pretext_pretrained_path is not None:
            knn_path = os.path.join(pretext_pretrained_path, os.path.basename(config_params['knn_val']))
        assert os.path.isfile(knn_path)
        indices = np.load(knn_path)
        dataset = NeighborsDataset(dataset, indices,
                                   num_neighbors=config_params['num_neighbors'],
                                   all_neighbors=config_params['all_neighbors'])  # Only use 5

    return dataset


def get_train_dataloader(config_param: dict, dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, num_workers=config_param['num_workers'],
                      batch_size=config_param['batch_size'], pin_memory=True, collate_fn=collate_custom,
                      drop_last=True, shuffle=True)


def get_val_dataloader(config_param: dict, dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, num_workers=config_param['num_workers'],
                      batch_size=config_param['batch_size'], pin_memory=True, collate_fn=collate_custom,
                      drop_last=False, shuffle=False)
