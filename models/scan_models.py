"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.utils import get_pyramid_patchs
from models.seq_blocks import MlpNet


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048

    else:
        raise NotImplementedError


def get_model(config_params: dict, pretrain_path: str = None) -> nn.Module:
    # Get backbone
    if config_params['backbone'] == 'resnet50':
        if 'imagenet' in config_params['train_db_name'] or 'flowers' in config_params['train_db_name']:
            from models.resnet import resnet50
            backbone = resnet50()

        else:
            raise NotImplementedError

    else:
        raise ValueError('Invalid backbone {}'.format(config_params['backbone']))

    # Setup
    if config_params['setup'] in ['simclr', 'moco']:
        model = ContrastiveModel(backbone, **config_params['model_kwargs'])

    elif config_params['setup'] == 'simclr-flowers':
        if config_params['model_kwargs']['use_att']:
            model = ContrastiveAttPatchModel(backbone=backbone,
                                             patch_size=config_params['patch_size'],
                                             step_scale=config_params['patch_overlap_scale'],
                                             **config_params['model_kwargs'])
        else:
            model = ContrastivePatchModel(backbone=backbone,
                                          patch_size=config_params['patch_size'],
                                          **config_params['model_kwargs'])

    elif config_params['setup'] in ['scan', 'selflabel']:
        if config_params['setup'] == 'selflabel':
            assert (config_params['num_heads'] == 1)
        model = ClusteringModel(backbone,
                                nclusters=config_params['num_classes'],
                                nheads=config_params['num_heads'],
                                heads_layers=config_params['heads_layers'],
                                norm_features=config_params['norm_features'],
                                contrastive_head=config_params['contrastive_head'],
                                contrastive_dim=config_params['contrastive_dim'],
                                from_contrastive=config_params['from_contrastive']
                                )

    else:
        raise ValueError('Invalid setup {}'.format(config_params['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        model_path = os.path.join(pretrain_path, os.path.basename(config_params['pretext_model']))
        assert os.path.isfile(model_path)
        print(f'Load pretrained weights from: {model_path}')
        state = torch.load(model_path, map_location='cpu')

        if config_params['setup'] == 'scan':  # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)
            # assert all(map(lambda x: x.split('.')[0] == 'cluster_head', missing.missing_keys))

        elif config_params['setup'] == 'selflabel':  # Weights are supposed to be transfered from scan
            # We only continue with the best head (pop all heads first, then copy back the best head)
            model_state = state['model']
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['cluster_head.%d.weight' % (state['head'])]
            best_head_bias = model_state['cluster_head.%d.bias' % (state['head'])]
            for k in all_heads:
                model_state.pop(k)

            model_state['cluster_head.0.weight'] = best_head_weight
            model_state['cluster_head.0.bias'] = best_head_bias
            missing = model.load_state_dict(model_state, strict=True)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim=1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone,
                 nclusters: int,
                 nheads=1,
                 heads_layers: int = 1,
                 norm_features: bool = False,
                 contrastive_head='mlp',
                 contrastive_dim=None,
                 from_contrastive: bool = False):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        self.norm_features = norm_features
        self.from_contrastive = from_contrastive
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)

        if contrastive_head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, contrastive_dim)

        elif contrastive_head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(),
                nn.Linear(self.backbone_dim, contrastive_dim))
        else:
            self.contrastive_head = None

        if heads_layers == 1:
            self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])
        else:
            if from_contrastive:
                in_ch_cluster = contrastive_dim
                multy_coeff = 0.5
                ch_list = [in_ch_cluster, in_ch_cluster, in_ch_cluster // 2, in_ch_cluster // 4, nclusters]
                heads_layers = len(ch_list) - 1
            else:
                in_ch_cluster = self.backbone_dim
                multy_coeff = 0.25
                ch_list = None
            self.cluster_head = nn.ModuleList([MlpNet(ch_list=ch_list,
                                                      in_ch=in_ch_cluster,
                                                      deep=heads_layers,
                                                      active_type='leakly_relu',
                                                      out_ch=nclusters,
                                                      multy_coeff=multy_coeff) for _ in range(self.nheads)])

    def forward_cluster_head(self, features):
        out = [cluster_head(features) for cluster_head in self.cluster_head]
        return out

    def forward_cluster_heads(self, features):
        out_clusters = []
        for cluster_head in self.cluster_head:
            out_ = cluster_head(features)
            if self.norm_features:
                out_ = nn.functional.normalize(out_, dim=-1, p=2)
            out_clusters.append(out_)
        return out_clusters

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_backbone_neighbors(self, x):
        batch_size, nbh, c, h, w = x.shape
        neighbors_backbone = self.forward_backbone(x.view(batch_size * nbh, c, h, w))
        neighbors_backbone = neighbors_backbone.view(batch_size, nbh, neighbors_backbone.shape[-1])
        return neighbors_backbone

    def forward_contrastive_head(self, features):
        contrastive_features = self.contrastive_head(features)
        contrastive_features = F.normalize(contrastive_features, dim=-1)
        return contrastive_features

    def forward(self, x, forward_all=False, neighbors=False):
        if neighbors:
            features_backbone = self.forward_backbone_neighbors(x)
        else:
            features_backbone = self.forward_backbone(x)

        features_contrastive = self.forward_contrastive_head(features_backbone)
        if self.from_contrastive:
            out_clusters = self.forward_cluster_heads(features_contrastive)
        else:
            out_clusters = self.forward_cluster_heads(features_backbone)
        output = {'out_clusters': out_clusters}
        if forward_all:
            output['features_backbone'] = features_backbone
            output['features_contrastive'] = features_contrastive
        return output


class ContrastivePatchModel(ContrastiveModel):
    def __init__(self, backbone, patch_size=32, head='mlp', features_dim=128, **kwargs):
        super(ContrastivePatchModel, self).__init__(backbone=backbone, head=head, features_dim=features_dim)
        self._patch_size = patch_size

    def forward(self, x_full_size: Tensor, x_patches: Tensor = None):
        if isinstance(x_patches, Tensor):
            b, p, c, h, w = x_patches.shape
            features_patches = [super(ContrastivePatchModel, self).forward(x_patches[:, ind]) for ind in range(p)]
            features_patches = torch.mean(torch.stack(features_patches, 0), dim=0)
        else:
            features_patches = super(ContrastivePatchModel, self).forward(x_full_size)
        return features_patches


class AttentionOneBlock(nn.Module):
    def __init__(self, no_softmax=False):
        super(AttentionOneBlock, self).__init__()
        # self.padding = 2
        self.no_softmax = no_softmax
        self.layers = \
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(8, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(8, 1, kernel_size=3, padding=(1, 1)),
            )

    def forward(self, x):
        x = self.layers(x)
        x_size = x.shape
        if self.no_softmax:
            return x.view(x_size)
        else:
            x = F.softmax(x.view(x.shape[0], -1), dim=1)
            return x.view(x_size)


class ContrastiveAttPatchModel(ContrastiveModel):
    def __init__(self, backbone, patch_size=32, step_scale=2, head='mlp', features_dim=128, **kwargs):
        super(ContrastiveAttPatchModel, self).__init__(backbone=backbone, head=head, features_dim=features_dim)
        self._patch_size = patch_size
        self._step_scale = step_scale
        self._att_net = AttentionOneBlock(no_softmax=False)

    def forward(self, x_full_size: Tensor, x_patches: Tensor = None):

        att_map_ful_size = self._att_net(x_full_size)

        if isinstance(x_patches, Tensor):
            b, p, c, h, w = x_patches.shape
            att_patches = get_pyramid_patchs(att_map_ful_size, patch_szie=self._patch_size, step_scale=self._step_scale)
            features_patches = [super(ContrastiveAttPatchModel, self).forward(att_patches[:, ind] * x_patches[:, ind])
                                for ind in range(p)]
            features_patches = torch.mean(torch.stack(features_patches, 0), dim=0)
        else:
            features_patches = super(ContrastiveAttPatchModel, self).forward(att_map_ful_size * x_full_size)
        return features_patches
