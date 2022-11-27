import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn


def get_optimizer(config_params: dict, model: nn.Module, cluster_head_only: bool = False) -> torch.optim.Optimizer:
    if config_params['setup'] == 'scan':
        if model.contrastive_head is not None:
            for name, param in model.contrastive_head.named_parameters():
                param.requires_grad = False
        if cluster_head_only:  # Only weights in the cluster head will be updated
            for name, param in model.cluster_head.named_parameters():
                param.requires_grad = True
            params = model.cluster_head.parameters()
        else:
            params = chain(model.backbone.parameters(), model.cluster_head.parameters())
    else:
        params = model.parameters()

    if config_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **config_params['optimizer_kwargs'])
    elif config_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **config_params['optimizer_kwargs'])
    else:
        raise ValueError('Invalid optimizer {}'.format(config_params['optimizer']))

    return optimizer


def adjust_learning_rate(config_param: dict, optimizer: torch.optim.Optimizer, epoch: int):
    lr = config_param['optimizer_kwargs']['lr']

    if config_param['scheduler'] == 'cosine':
        eta_min = lr * (config_param['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / config_param['epochs'])) / 2

    elif config_param['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(config_param['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (config_param['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif config_param['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(config_param['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
