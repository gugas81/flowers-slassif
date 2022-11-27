import os
import time

import fire
import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import create_config
from data import get_train_dataset, get_val_dataset, get_train_dataloader, get_val_dataloader, \
    get_train_transformations, get_val_transformations
from losses import get_criterion
from models import get_model
from trainer.memory import MemoryBank, fill_memory_bank
from trainer.model_evaluate import contrastive_evaluate
from trainer.optimizers import get_optimizer, adjust_learning_rate


def simclr_train(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    model.train()

    for i, batch in tqdm(enumerate(train_loader)):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        if 'image_patches' in batch:
            images_patches = batch['image_patches'].cuda(non_blocking=True)
            images_augmented_patches = batch['image_augmented_patches'].cuda(non_blocking=True)
            output = model(images, images_patches)
            output_augmented = model(images_augmented, images_augmented_patches)
        else:
            output = model(images)
            output_augmented = model(images_augmented)
        output_ = torch.cat([output.unsqueeze(1), output_augmented.unsqueeze(1)], dim=1)

        # output = model(input_).view(b, 2, -1)
        loss = criterion(output_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 10 == 0:
        #     progress.display(i)


def run_trainer(config_file_path: str,
                config_exp_path: str,
                exp_name: str = None,
                pretrained_ckpt: str = None,
                dbg_mode: bool = False):
    # Retrieve common file
    p = create_config(config_file_path, config_exp_path, exp_name)
    # p['epochs'] = 165
    if dbg_mode:
        p['batch_size'] = 8
        p['num_workers'] = 0
    print(colored(p, 'magenta'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                      split='train+unlabeled')  # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train')  # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset)

    np.save(p['class_to_idx_tr'], base_dataset.class_to_idx)
    np.save(p['class_to_idx_val'], val_dataset.class_to_idx)

    # if p['train_db_name'] == 'flowers-data':
    #     memory_bank_base = MemoryBank(len(val_dataset),
    #                             p['model_kwargs']['features_dim'],
    #                             p['num_classes'], p['criterion_kwargs']['temperature'])
    # else:
    memory_bank_base = MemoryBank(len(base_dataset),
                                  p['model_kwargs']['features_dim'],
                                  p['num_classes'], p['criterion_kwargs']['temperature'])

    memory_bank_base.cuda()

    memory_bank_val = MemoryBank(len(val_dataset),
                                 p['model_kwargs']['features_dim'],
                                 p['num_classes'], p['criterion_kwargs']['temperature'],
                                 knn=8)
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    pretrained_ckpt = pretrained_ckpt if pretrained_ckpt is not None else p['pretext_checkpoint']
    if os.path.exists(pretrained_ckpt):
        print(f'Load checkpoint from: {pretrained_ckpt}')
        print(colored(f'Restart from checkpoint {pretrained_ckpt}', 'blue'))
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']
        print(f'training will continue from: epoch={start_epoch}')

    else:
        print(colored(f'No checkpoint file at {pretrained_ckpt}', 'blue'))
        start_epoch = 0
        model = model.cuda()

        # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Evaluate (To monitor progress - Not for validation)
    print('Zero Evaluate ...')
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    top1_tr, knn_tr, wiegths_knn_tr = contrastive_evaluate(base_dataloader, model, memory_bank_base)
    print('Result of kNN evaluation(val) is %.2f' % (top1_tr))
    # if p['train_db_name'] == 'flowers-data':
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    # p['fist_batch_val_path']

    np.save(p['features_tr_path'], memory_bank_val.features.cpu().numpy())
    np.save(p['features_val_path'], memory_bank_base.features.cpu().numpy())

    np.save(p['targets_val_path'], memory_bank_val.targets.cpu().numpy())
    np.save(p['targets_tr_path'], memory_bank_base.targets.cpu().numpy())

    # else:
    #     fill_memory_bank(base_dataloader, model, memory_bank_base)
    top1_val, knn_val, wiegths_knn_val = contrastive_evaluate(val_dataloader, model, memory_bank_val)
    print('Result of kNN evaluation(val) is %.2f' % (top1_val))
    # np.save(p['wiegths_knn_val'], wiegths_knn_val)

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        t_start = time.time()
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)

        fill_memory_bank(base_dataloader, model, memory_bank_base)
        top1_tr, knn_tr, wiegths_knn_tr = contrastive_evaluate(base_dataloader, model, memory_bank_base)
        print(f'Result Train of kNN evaluation epoch {epoch} is {top1_tr: .2f}')
        print(f"Save KNN inds in: {p['knn_train']}")
        np.save(p['knn_train'], knn_tr)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        # if p['train_db_name'] == 'flowers-data':
        fill_memory_bank(val_dataloader, model, memory_bank_val)
        # else:
        #     fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...val train set')
        top1_val, knn_val, wiegths_knn_val = contrastive_evaluate(val_dataloader, model, memory_bank_val)
        print(f'Result Validation of kNN evaluation epoch {epoch} is {top1_val: .2f}')
        print(f"Save KNN inds in: {p['knn_val']}")
        np.save(p['knn_val'], knn_val)

        # Checkpoint
        print(f"Checkpoint saving in {p['pretext_checkpoint']}")
        torch.save({'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'epoch': epoch + 1},
                   p['pretext_checkpoint'])

        time_eval = time.time() - t_start
        print(f'Eval time of epoch {epoch} is {time_eval / 60:.2f} min')

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Train data: Mine the nearest neighbors (Top-%d)' % (topk))
    indices_tr, acc_tr = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * acc_tr))
    np.save(p['topk_neighbors_train_path'], indices_tr)

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    if p['train_db_name'] == 'flowers-data':
        fill_memory_bank(val_dataloader, model, memory_bank_base)
    else:
        fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices_val, acc_val = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' % (topk, 100 * acc_val))
    np.save(p['topk_neighbors_val_path'], indices_val)


if __name__ == '__main__':
    fire.Fire(run_trainer)
