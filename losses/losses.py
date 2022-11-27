"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8


def get_criterion(criterion_type: str, reduce=True, **criterion_kwargs):
    if criterion_type == 'simclr':
        criterion = SimCLRLoss(**criterion_kwargs)

    elif criterion_type == 'scan':
        criterion = SCANLoss(**criterion_kwargs)

    elif criterion_type == 'scan-weighted':
        criterion = SCANWeightedLoss(reduce=reduce, **criterion_kwargs)

    elif criterion_type == 'confidence-cross-entropy':
        criterion = ConfidenceBasedCE(**criterion_kwargs)

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, x_input, x_target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(x_target, mask)
        b, c = x_input.size()
        n = target.size(0)
        input = torch.masked_select(x_input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0, reduce=True, **kwargs):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.bce = nn.BCELoss(reduce=reduce)
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


class SCANWeightedLoss(SCANLoss):
    def __init__(self, entropy_weight=2.0, temperature=1.0, reduce=True, **kwargs):
        super(SCANWeightedLoss, self).__init__(entropy_weight=entropy_weight, reduce=reduce, **kwargs)
        self.temperature = temperature

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        anchors_prob = self.softmax(anchors['out_clusters'])
        neighbors_prob = self.softmax(neighbors['out_clusters'])
        contrastive_features_anchor = anchors['contrastive_features']
        contrastive_features_neighbors = neighbors['contrastive_features']

        consistency_loss, _ = self.get_consistency_loss(anchors_prob,
                                                     neighbors_prob,
                                                     contrastive_features_anchor,
                                                     contrastive_features_neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss

    def get_consistency_loss(self,
                             anchors_prob,
                             neighbors_prob,
                             contrastive_features_anchor,
                             contrastive_features_neighbors):
        weights_neighbors = self.get_weights_neighbors(contrastive_features_anchor, contrastive_features_neighbors)
        neighbors_weighted_prob = torch.sum(weights_neighbors.unsqueeze(-1) * neighbors_prob, 1)
        similarity = torch.bmm(anchors_prob.unsqueeze(-2), neighbors_weighted_prob.unsqueeze(-1)).squeeze()
        ones = torch.ones_like(similarity)

        consistency_loss = self.bce(similarity, ones)
        return consistency_loss, weights_neighbors

    def get_weights_neighbors(self, features_anchor, features_neighbors):
        nbh = features_neighbors.size(1)
        correlation_features = []
        for ind in range(nbh):
            corr = torch.bmm(features_anchor.unsqueeze(-2),
                             features_neighbors[:, ind].unsqueeze(-1))
            corr = corr.squeeze().div_(self.temperature)
            correlation_features.append(corr)
        correlation_features = torch.stack(correlation_features, 1)  # L2 distance
        weights_neighbors = self.softmax(correlation_features)
        return weights_neighbors


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + torch.finfo(exp_logits.dtype).eps)
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss
