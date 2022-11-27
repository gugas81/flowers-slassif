"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature, knn: int = 100):
        self.n = n
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.knn = knn
        self.temperature = temperature
        self.num_classes = num_classes
        # self.consistency_loss = SCANWeightedLoss(temperature=self.temperature)

    def weighted_knn(self, features_batch):
        # perform weighted knn
        # retrieval_one_hot = torch.zeros(self.knn, self.num_classes).to(self.device)
        batch_size = features_batch.shape[0]
        simmularuty = torch.matmul(features_batch, self.features.t())
        sim_knn, knn_ind = simmularuty.topk(self.knn, dim=1, largest=True, sorted=True)
        sim_knn, knn_ind = sim_knn[:, 1:], knn_ind[:, 1:]
        targets = self.targets.view(1, -1).expand(batch_size, -1)
        targets_knn = torch.gather(targets, 1, knn_ind)
        # retrieval_one_hot.resize_(batchSize * self.knn, self.num_classes).zero_()
        # retrieval_one_hot.scatter_(1, targets_knn.view(-1, 1), 1)
        # yd_transform = sim_knn.clone().div_(self.temperature).exp_()
        # features_neighbors = torch.stack([self.features[knn_ind[ind_]] for ind_ in range(knn_ind.shape[0])], 0)
        targets_knn_one_hot = F.one_hot(targets_knn, num_classes=self.num_classes)
        wiegths_knn = F.softmax(sim_knn.clone().div_(self.temperature), -1)
        probs = torch.sum(torch.mul(targets_knn_one_hot, wiegths_knn.unsqueeze(-1)), 1)
        #                             wiegths_knn.view(batchSize, -1, 1)), 1)
        # probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, self.num_classes),
        #                             wiegths_knn.view(batchSize, -1, 1)), 1)
        # _, class_preds = probs.sort(1, True)
        class_pred = torch.argmax(probs, 1)  # class_preds[:, 0]

        return class_pred, knn_ind, wiegths_knn

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk + 1)  # Sample itself is included

        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy

        else:
            return indices

    def reset(self):
        self.ptr = 0

    def update(self, features, targets):
        b = features.size(0)

        assert (b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr + b].copy_(features.detach())
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader: DataLoader, model: nn.Module, memory_bank: MemoryBank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        if 'image_patches' in batch:
            image_patches = batch['image_patches'].cuda(non_blocking=True)
            output = model(images, image_patches)
        else:
            output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' % (i, len(loader)))
