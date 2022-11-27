import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch._six import string_classes


def get_pathes_half_overlap(img, patch_szie, step_scale=2):
    if img.dim() < 4:
        no_batch = True
        img = img[None]
    else:
        no_batch = False

    img_patches = img.unfold(-1, patch_szie, patch_szie // step_scale).unfold(-3, patch_szie, patch_szie // step_scale)
    b, c, p1, p2, h, w = img_patches.shape
    img_patches = img_patches.reshape(b, c, p1 * p2, h, w).permute((0, 2, 1, 3, 4))

    if no_batch:
        img_patches = img_patches[0]

    return img_patches


def get_pyramid_patchs(img, patch_szie, step_scale=2):
    with_batch = img.dim() == 4
    img_size = [patch_szie, patch_szie]
    # resize_to_patch = tf.Resize([patch_szie, patch_szie])
    img_0 = img.unsqueeze(0) if not with_batch else img
    img_0 = F.interpolate(img_0, size=img_size, mode='bilinear').unsqueeze(-4)
    img_1_pathes = get_pathes_half_overlap(img, patch_szie * 2)
    if with_batch:
        b, p, c, h, w = img_1_pathes.shape
        img_1_pathes = img_1_pathes.reshape(b, c * p, h, w)
    else:
        img_0 = img_0[0]
    img_1_pathes = F.interpolate(img_1_pathes, size=img_size, mode='bilinear')
    if with_batch:
        img_1_pathes = img_1_pathes.reshape(b, p, c, patch_szie, patch_szie)
    img_2_pathes = get_pathes_half_overlap(img, patch_szie, step_scale)
    #     print(img_0.shape, img_1_pathes.shape, img_2_pathes.shape)
    return torch.cat([img_0, img_1_pathes, img_2_pathes], -4)


""" Custom collate function """


def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))
