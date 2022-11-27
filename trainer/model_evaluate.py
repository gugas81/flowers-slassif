import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config_params
from data import NeighborsDataset
from data import get_val_dataloader, get_val_dataset, get_val_transformations
from losses import entropy
from losses import get_criterion
from models import get_model
from trainer.memory import MemoryBank


def scan_eval(config_file_path: str, pretrained_ckpt: str):
    config_param = get_config_params(config_file_path)
    val_transformations = get_val_transformations(config_param)

    val_dataset = get_val_dataset(config_param, val_transformations, to_neighbors_dataset=True,
                                  pretext_pretrained_path=pretrained_ckpt)

    val_dataloader = get_val_dataloader(config_param, val_dataset)
    print('Validation transforms:', val_transformations)

    print(f'Pretrained pretext_model will be loaded from: {pretrained_ckpt}')
    model = get_model(config_param, pretrained_ckpt)
    print(model)

    criterion_val = get_criterion(config_param, reduce=False)
    criterion_val.cuda()

    # Evaluate
    print('Make prediction on validation set ...')
    predictions_val, losses_val, features_anchor_val, features_neighbors_val = \
        get_scan_predictions(config_param, val_dataloader, model, criterion_val)
    print(f'Losses  val: {losses_val}')

    clustering_stats = hungarian_evaluate(predictions_val, compute_confusion_matrix=False)
    print(clustering_stats)


@torch.no_grad()
def contrastive_evaluate(val_loader: DataLoader, model: nn.Module, memory_bank: MemoryBank):
    model.eval()
    knn_inds = []
    wiegths_knn = []
    acc = []
    for batch in val_loader:
        # with torch.autograd.detect_anomaly():
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)
        if 'image_patches' in batch:
            images_patches = batch['image_patches'].cuda(non_blocking=True)
            output = model(images, images_patches)
        else:
            output = model(images)

        pred_knn, knn_ind_batch, wiegths_knn_batch = memory_bank.weighted_knn(output)
        #
        knn_inds.append(knn_ind_batch)
        wiegths_knn.append(wiegths_knn_batch)
        acc_curr = 100 * torch.mean(torch.eq(pred_knn, target).float())
        acc.append(float(acc_curr.cpu().numpy()))
    knn_inds = torch.cat(knn_inds, 0).cpu().numpy()
    wiegths_knn = torch.cat(wiegths_knn, 0).cpu().numpy()
    acc_avg = np.array(acc).mean()
    return acc_avg, knn_inds, wiegths_knn


@torch.no_grad()
def get_scan_predictions(config_params: dict, dataloader: DataLoader, model: nn.Module, criterion: nn.Module):
    # Make predictions on a dataset with neighbors
    model.eval()
    criterion.eval()
    predictions_anchors = []
    predictions_anchors_nbh = []
    probs_anchors = []
    targets_anchors = []
    features_anchor = []
    consistency_losses = []
    n_classes = config_params['num_classes']

    scan_mode = (config_params['setup'] == 'scan')
    if scan_mode:
        features_name = 'features_contrastive'
    else:
        features_name = 'features'
    #
    # features_contrastive = torch.tensor(np.load(p['features_val_path'])).cuda()
    # knn_val = torch.tensor(np.load(p['knn_val'])).cuda()
    # wiegths_knn_val = torch.tensor(np.load(p['wiegths_knn_val'])).cuda()
    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors_ind = []
        predictions_nbh = []
        prob_nbh = []
        features_neighbors = []
    else:
        key_ = 'image'
        include_neighbors = False

    for batch in tqdm(dataloader):
        # possible_neighbors = batch['possible_neighbors']
        images_anchor = batch[key_].cuda(non_blocking=True)

        res_anchor = model(images_anchor, forward_all=True)
        anchors_output = res_anchor['out_clusters'][0]
        features_anchor_batch = res_anchor[features_name]
        anchors_pred = torch.argmax(anchors_output, dim=-1)
        anchors_prob = F.softmax(anchors_output, dim=-1)

        neighbors = batch['neighbors'].cuda(non_blocking=True)
        neighbors_res = model(neighbors, forward_all=True, neighbors=True)
        neighbors_output = neighbors_res['out_clusters'][0]
        features_neighbors_batch = neighbors_res[features_name]
        neighbors_prob = F.softmax(neighbors_output, dim=-1)
        neighbors_pred = torch.argmax(neighbors_output, dim=-1)
        neighbors_targets = batch['target_neighbors'].cuda(non_blocking=True)

        consistency_loss, weights_neighbors = criterion.get_consistency_loss(anchors_prob,
                                                                             neighbors_prob,
                                                                             features_anchor_batch,
                                                                             features_neighbors_batch)

        anchors_prob_nbh = (weights_neighbors.unsqueeze(-1) * F.one_hot(neighbors_targets, num_classes=n_classes)).sum(
            1)
        anchors_ped_nbh = torch.argmax(anchors_prob_nbh, -1)

        features_anchor.append(features_anchor_batch.cpu())
        predictions_anchors.append(anchors_pred.cpu())
        probs_anchors.append(anchors_prob.cpu())
        targets_anchors.append(batch['target'])

        # if include_neighbors:
        neighbors_ind.append(batch['possible_neighbors'].cpu())
        features_neighbors.append(features_neighbors_batch.cpu())
        predictions_nbh.append(neighbors_pred.cpu())
        prob_nbh.append(neighbors_prob.cpu())
        predictions_anchors_nbh.append(anchors_ped_nbh.cpu())
        consistency_losses.append(consistency_loss.cpu())

    predictions_anchors = torch.cat(predictions_anchors, dim=0)
    predictions_nbh = torch.cat(predictions_nbh, dim=0)
    probs_anchors = torch.cat(probs_anchors, dim=0)
    prob_nbh = torch.cat(prob_nbh, dim=0)
    targets_anchors = torch.cat(targets_anchors, dim=0)
    predictions_anchors_nbh = torch.cat(predictions_anchors_nbh, dim=0)
    consistency_losses = torch.cat(consistency_losses, dim=0)

    out = {'predictions_anchors': predictions_anchors,
           'predictions_nbh': predictions_nbh,
           'probabilities_anchors': probs_anchors,
           'prob_nbh': prob_nbh,
           'targets_anchors': targets_anchors,
           'consistency_losses': consistency_losses,
           'predictions_anchors_nbh': predictions_anchors_nbh}

    features_anchor = torch.cat(features_anchor, dim=0)
    features_neighbors = torch.cat(features_neighbors, dim=0)
    entropy_loss = entropy(probs_anchors, input_as_probabilities=True)
    acc_pred_nbh = (predictions_anchors_nbh == targets_anchors).type_as(probs_anchors).mean()
    losses = {'entropy_loss': entropy_loss,
              'consistency_losses': consistency_losses.mean(),
              'acc_pred_nbh': acc_pred_nbh}
    return out, losses, features_anchor, features_neighbors


@torch.no_grad()
def hungarian_evaluate(pred_out: dict, class_names=None,
                       compute_purity=True, compute_confusion_matrix=True,
                       confusion_matrix_file=None) -> dict:
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    # head = all_predictions[subhead_index]
    targets = pred_out['targets_anchors'].cuda()
    predictions = pred_out['predictions_anchors'].cuda()
    probs = pred_out['probabilities_anchors'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    reordered_prob = torch.zeros_like(probs).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = float((reordered_preds == targets).type_as(reordered_prob).mean().cpu().numpy())
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    stat_classes_pred = {f'class_{ind_}': float((reordered_preds == ind_).type_as(reordered_prob).mean().cpu().numpy())
                         for ind_ in range(num_classes)}
    stat_classes_gt = {f'class_{ind_}': float((targets == ind_).type_as(reordered_prob).mean().cpu().numpy())
                       for ind_ in range(num_classes)}

    stat_classes_err = {
        f'class_{ind_}': float((reordered_preds[targets == ind_] == ind_).type_as(reordered_prob).mean().cpu().numpy())
        for ind_ in range(num_classes)}

    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
        reordered_prob[..., target_i] = probs[..., pred_i]
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # # Compute confusion matrix
    # if compute_confusion_matrix:
    #     confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
    #                         class_names, confusion_matrix_file)
    cross_entropy = F.cross_entropy(reordered_prob, F.one_hot(targets, num_classes=num_classes).type_as(reordered_prob))
    cross_entropy = float(cross_entropy.cpu().numpy())

    return {'ACC': acc,
            'cross_entropy': cross_entropy,
            'ARI': ari, 'NMI': nmi,
            'ACC Top-5': top5,
            'hungarian_match': match,
            'stat_classes_pred': stat_classes_pred,
            'stat_classes_gt': stat_classes_gt,
            'stat_classes_err': stat_classes_err}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


if __name__ == '__main__':
    fire.Fire()
