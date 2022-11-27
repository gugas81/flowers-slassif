# List of augmentations based on randaugment
import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

random_mirror = True


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'flowers':
        # Augmentation strategy from the SimCLR paper

        return transforms.Compose([
            # SquarePad(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomApply([
            #     transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            # ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            # transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'ours':
        # Augmentation strategy from our paper
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes=p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length=p['augmentation_kwargs']['cutout_kwargs']['length'],
                random=p['augmentation_kwargs']['cutout_kwargs']['random'])])

    elif p['augmentation_strategy'] == 'scan-flowers':
        # Augmentation strategy from our paper
        return transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            # Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
            # Cutout(
            #     n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
            #     length = p['augmentation_kwargs']['cutout_kwargs']['length'],
            #     random = p['augmentation_kwargs']['cutout_kwargs']['random'])]
        ])

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    if p['train_db_name'] == 'flowers-data':
        if p['setup'] == 'scan':
            return transforms.Normalize(**p['transformation_kwargs']['normalize'])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    else:
        crop_size = p['transformation_kwargs']['crop_size']
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])


def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Identity(img, v):
    return img


def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def augment_list():
    l = [
        (Identity, 0, 1),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


class Augment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class Cutout(object):
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def generate_aug_params(aug_kwargs: dict, img_size):
    aff_params = aug_kwargs['affine']
    angle, translations, scale, shear = transforms.RandomAffine.get_params(img_size=img_size, **aff_params)
    aug_param = {'hflip': torch.rand(1) < aug_kwargs['p_flip'],
                 'vflip': torch.rand(1) < aug_kwargs['p_flip'],
                 'color_jitter_apply': torch.rand(1) < aug_kwargs['color_jitter_random_apply']['p'],
                 'color_jitter_params': aug_kwargs['color_jitter'],
                 'gray_scale': torch.rand(1) < aug_kwargs['random_grayscale']['p'],
                 'apply_affine': torch.rand(1) < aug_kwargs['p_affine'],
                 'affine': {'angle': angle,
                            'translate': translations,
                            'scale': scale,
                            'shear': shear}
                 }
    return aug_param


def apply_aug(img, aug_param):
    if aug_param['hflip']:
        img = tf.hflip(img)
    if aug_param['vflip']:
        img = tf.vflip(img)
    if aug_param['color_jitter_apply']:
        img = transforms.ColorJitter(**aug_param['color_jitter_params'])(img)
    if aug_param['gray_scale']:
        img = tf.rgb_to_grayscale(img, num_output_channels=img.shape[-3])
    if aug_param['apply_affine']:
        img = tf.affine(img, **aug_param['affine'])
    return img
