"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import generate_aug_params, apply_aug

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""


class AugmentedDataset(Dataset):
    def __init__(self, dataset, aug_kwargs=None):
        super(AugmentedDataset, self).__init__()
        if aug_kwargs is None:
            transform = dataset.transform
            dataset.transform = None
            if isinstance(transform, dict):
                self.image_transform = transform['standard']
                self.augmentation_transform = transform['augment']

            else:
                self.image_transform = transform
                self.augmentation_transform = transform
        self.dataset = dataset
        self.aug_kwargs = aug_kwargs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_item = self.dataset.__getitem__(index)
        image = image_item['image']

        if self.aug_kwargs is None:
            image_item['image'] = self.image_transform(image)
            image_item['image_augmented'] = self.augmentation_transform(image)
        else:
            aug_params = generate_aug_params(self.aug_kwargs, image.size())
            image_item['image_augmented'] = apply_aug(image, aug_params)
            image_item['meta']['aug_params'] = aug_params
        return image_item


class AugmentedPyramidDataset(Dataset):
    def __init__(self, dataset, aug_kwargs):
        super(AugmentedPyramidDataset, self).__init__()
        self.dataset = dataset
        self.aug_kwargs = aug_kwargs

    def __len__(self):
        return len(self.dataset)

    @property
    def targets(self):
        if self.no_labels:
            return None
        else:
            return self.dataset.targets

    @property
    def class_to_idx(self):
        if self.no_labels:
            return None
        else:
            return self.dataset.class_to_idx

    def __getitem__(self, index):
        image_item = self.dataset.__getitem__(index)
        aug_params = generate_aug_params(self.aug_kwargs)
        image_item['image_augmented'] = apply_aug(image_item['image'], aug_params)
        image_item['image_augmented_patches'] = self.dataset.get_pyramid_patches(image_item['image_augmented'])
        image_item['meta']['aug_params'] = aug_params
        return image_item


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""


class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None, aug_kwargs=None, all_neighbors: bool = False):
        super(NeighborsDataset, self).__init__()
        self.aug_kwargs = aug_kwargs
        self.all_neighbors = all_neighbors
        if aug_kwargs is None:
            transform = dataset.transform
            if isinstance(transform, dict):
                self.anchor_transform = transform['standard']
                self.neighbor_transform = transform['augment']
            else:
                self.anchor_transform = transform
                self.neighbor_transform = transform

            dataset.transform = None
        self.dataset = dataset
        self.indices_neighbors = indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices_neighbors = self.indices_neighbors[:, :num_neighbors + 1]
            self.num_neighbors = min(num_neighbors, self.indices_neighbors.shape[-1])
        else:
            self.num_neighbors = self.indices_neighbors.shape[-1]
        assert (self.indices_neighbors.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    @property
    def targets(self):
        return self.dataset.targets

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        nbh_ids = self.indices_neighbors[index]
        if self.all_neighbors:
            neighbors_images = []
            neighbors_targets = []
            for ind in nbh_ids[1:]:
                nbh_item = self.dataset[ind]
                neighbors_images.append(nbh_item['image'])
                neighbors_targets.append(nbh_item['target'])
            neighbors_images = torch.stack(neighbors_images)
            neighbors_targets = torch.tensor(neighbors_targets)
        else:
            neighbor_id = np.random.choice(self.indices_neighbors[nbh_ids], 1)[0]
            neighbors_images = self.dataset[neighbor_id]['image']
        output['meta'] = anchor['meta']

        if self.aug_kwargs is None:
            anchor['image'] = self.anchor_transform(anchor['image'])
            neighbors_images = self.neighbor_transform(neighbors_images)
        else:
            aug_params_anchor = generate_aug_params(self.aug_kwargs, anchor['image'].size())
            aug_params_neighbor = generate_aug_params(self.aug_kwargs, anchor['image'].size())
            anchor['image'] = apply_aug(anchor['image'], aug_params_anchor)
            neighbors_images = apply_aug(neighbors_images, aug_params_neighbor)
            output['meta']['aug_params_anchor'] = aug_params_anchor
            output['meta']['aug_params_neighbor'] = aug_params_neighbor

        output['anchor'] = anchor['image']
        output['neighbors'] = neighbors_images
        output['possible_neighbors'] = torch.from_numpy(nbh_ids)
        output['target'] = anchor['target']
        output['target_neighbors'] = neighbors_targets
        return output
