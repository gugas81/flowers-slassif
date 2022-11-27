import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from PIL import Image
from torchvision import transforms as tf

from data.data_paths import FLOWERS_DS_UNLABELED
from .utils import get_pyramid_patchs


class FlowersDatasets(datasets.ImageFolder):
    def __init__(self, root: str = FLOWERS_DS_UNLABELED,
                 split='train',
                 transform=None,
                 img_size=None,
                 splitted: bool = True,
                 to_tensor: bool = True,
                 no_labels: bool = False):
        self.ds_path = os.path.join(root, split) if splitted else root
        super(FlowersDatasets, self).__init__(root=self.ds_path, transform=None)
        self.transform = transform
        self.split = split
        self.no_labels = no_labels

        if img_size is not None:
            self.resize = tf.Resize([img_size, img_size])
        else:
            self.resize = nn.Identity()
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.no_labels:
            target = torch.tensor(float('nan')).to(device='cpu')

        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)
        elif self.to_tensor:
            img = tf.ToTensor()(img).to(device='cpu')

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'path': path}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img


class FlowersDatasetsPyramidImgs(FlowersDatasets):
    def __init__(self,
                 root: str = FLOWERS_DS_UNLABELED,
                 split='train',
                 splitted: bool = True,
                 patch_size: int = 32,
                 step_scale: int = 2,
                 transform=None,
                 img_size=None,
                 to_tensor: bool = True,
                 no_labels: bool = False):
        super(FlowersDatasetsPyramidImgs, self).__init__(root=root,
                                                         transform=transform,
                                                         split=split,
                                                         img_size=img_size,
                                                         splitted=splitted,
                                                         to_tensor=to_tensor,
                                                         no_labels=no_labels)
        self.patch_size = patch_size
        self.step_scale = step_scale

    def get_pyramid_patches(self, img):
        return get_pyramid_patchs(img, self.patch_size, step_scale=self.step_scale)

    def __getitem__(self, index):
        # datasets.ImageFolder.__getitem__(self, index)
        image_item = super(FlowersDatasetsPyramidImgs, self).__getitem__(index)
        image_item['image_patches'] = self.get_pyramid_patches(image_item['image'])
        image_item['meta']['patch_size'] = self.patch_size
        return image_item
