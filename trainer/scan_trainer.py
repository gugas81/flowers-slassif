"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import fire
import torch
import torch.nn as nn
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import create_config
from data import get_train_dataset, get_val_dataset, get_train_dataloader, get_val_dataloader, \
    get_train_transformations, get_val_transformations
from losses import get_criterion
from models import get_model
from trainer.model_evaluate import get_scan_predictions, hungarian_evaluate
from trainer.optimizers import get_optimizer, adjust_learning_rate


def scan_train(train_loader: DataLoader,
               model: nn.Module,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               epoch: int,
               config_param: dict):
    """
    Train w/ SCAN-Loss
    """
    # total_losses = AverageMeter('Total Loss', ':.4e')
    # consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    # entropy_losses = AverageMeter('Entropy', ':.4e')
    # progress = ProgressMeter(len(train_loader),
    #     [total_losses, consistency_losses, entropy_losses],
    #     prefix="Epoch: [{}]".format(epoch))

    model.cluster_head.train()
    if model.contrastive_head is not None:
        model.contrastive_head.eval()
    if config_param['update_cluster_head_only']:
        model.backbone.eval()  # Update BN
    else:
        model.backbone.train()

    heads = model.nheads
    assert heads == 1

    for i, batch in tqdm(enumerate(train_loader)):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbors'].cuda(non_blocking=True)
        batch_size, nbh, c, h, w = neighbors.shape

        if config_param['update_cluster_head_only']:  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_backbone = model.forward_backbone(anchors)
                anchors_contrastive = model.forward_contrastive_head(anchors_backbone)
                neighbors_backbone = model.forward_backbone_neighbors(neighbors)
                neighbors_contrastive = model.forward_contrastive_head(neighbors_backbone)

            if config_param['from_contrastive']:
                anchors_clusters = model.forward_cluster_heads(anchors_contrastive)
                neighbors_clusters = model.forward_cluster_heads(neighbors_contrastive)
            else:
                anchors_clusters = model.forward_cluster_heads(anchors_backbone)
                neighbors_clusters = model.forward_cluster_heads(neighbors_backbone)

            anchors_output = {'features_backbone': anchors_backbone,
                              'contrastive_features': anchors_contrastive,
                              'out_clusters': anchors_clusters[0]}

            neighbors_output = {'features_backbone': neighbors_backbone,
                                'contrastive_features': neighbors_contrastive,
                                'out_clusters': neighbors_clusters[0]}

        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors, forward_pass='return_all')
            neighbors_output = model(neighbors.view(batch_size * nbh, c, h, w), forward_pass='return_all')

        # Loss for every head
        # total_loss, consistency_loss, entropy_loss = [], [], []
        total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
        # for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
        #     total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
        #                                                                  neighbors_output_subhead)
        #     total_loss.append(total_loss_)
        #     consistency_loss.append(consistency_loss_)
        #     entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        # total_losses.update(np.mean([v.item() for v in total_loss]))
        # consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        # entropy_losses.update(np.mean([v.item() for v in entropy_loss]))
        #
        # total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"total_loss={total_loss.detach().cpu().numpy():.4f}, "
                  f"consistency_loss={consistency_loss.detach().cpu().numpy():.4f}, "
                  f"entropy_loss={entropy_loss.detach().cpu().numpy():.4f}")
            # progress.display(i)


def run_scan_trainer(config_file_path: str,
                     config_exp_path: str,
                     exp_name: str = None,
                     pretrained_ckpt: str = None,
                     dbg_mode: bool = False):
    # args = FLAGS.parse_args()
    p = create_config(config_file_path, config_exp_path, exp_name=exp_name)

    if dbg_mode:
        p['batch_size'] = 32
        p['num_workers'] = 0
        # p['num_neighbors'] = 4
    print(colored(p, 'magenta'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations,
                                      split='train', to_neighbors_dataset=True,
                                      pretext_pretrained_path=pretrained_ckpt)
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset=True,
                                  pretext_pretrained_path=pretrained_ckpt
                                  )
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' % (len(train_dataset), len(val_dataset)))

    # Model
    print(colored('Get model', 'blue'))
    pretrained_ckpt = pretrained_ckpt if pretrained_ckpt is not None else p['pretext_model']
    print(f'Pretrained pretext_model will be loaded from: {pretrained_ckpt}')
    model = get_model(p, pretrained_ckpt)
    print(model)

    if p['multi_gpu']:
        print('multi_gpu')
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)
    criterion_val = get_criterion(p, reduce=False)
    criterion_val.cuda()

    # Checkpoint
    if os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']

    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None

    # Main loop
    print(colored('Starting main loop', 'blue'))

    # Evaluate
    print('Make prediction on validation set ...')
    predictions_val, losses_val, features_anchor_val, features_neighbors_val = \
        get_scan_predictions(p, val_dataloader, model, criterion_val)
    print(f'Losses  val: {losses_val}')

    # print('Evaluate based on SCAN loss ...')
    # scan_stats = scan_evaluate(predictions_val)
    # print(scan_stats)
    print('Evaluate with hungarian matching algorithm ...')
    # lowest_loss_head = scan_stats['lowest_loss_head']
    # lowest_loss = scan_stats['lowest_loss']
    # if lowest_loss < best_loss:
    #     print('New lowest loss on validation set: %.4f -> %.4f' % (best_loss, lowest_loss))
    #     print('Lowest loss head is %d' % (lowest_loss_head))
    #     best_loss = lowest_loss
    #     best_loss_head = lowest_loss_head
    clustering_stats = hungarian_evaluate(predictions_val, compute_confusion_matrix=False)
    print(clustering_stats)

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch + 1, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        t_start = time.time()
        scan_train(train_dataloader, model, criterion, optimizer, epoch, config_param=p)
        time_eval = time.time() - t_start
        print(f'Train eval time of epoch {epoch} is {time_eval / 60:.2f} min')

        # Evaluate 
        print('Make prediction on validation set ...')
        predictions_val, losses_val, features_anchor_val, features_neighbors_val = \
            get_scan_predictions(p, val_dataloader, model, criterion_val)
        print(f'Losses  val: {losses_val}')

        # print('Evaluate based on SCAN loss ...')
        # scan_stats = scan_evaluate(predictions_val)
        # print(scan_stats)
        # lowest_loss_head = scan_stats['lowest_loss_head']
        # lowest_loss = scan_stats['lowest_loss']
        # if lowest_loss < best_loss:
        #     print('New lowest loss on validation set: %.4f -> %.4f' % (best_loss, lowest_loss))
        #     print('Lowest loss head is %d' % (lowest_loss_head))
        #     best_loss = lowest_loss
        #     best_loss_head = lowest_loss_head
        #
        # if lowest_loss < best_loss:
        #     print('New lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
        #     print('Lowest loss head is %d' %(lowest_loss_head))
        #     best_loss = lowest_loss
        #     best_loss_head = lowest_loss_head
        #     print(f"Saving model epoch: {epoch}, to {p['scan_model']}")
        #     model_weights = model.module.state_dict() if p['multi_gpu'] else model.state_dict()
        #     torch.save({'model': model_weights, 'head': best_loss_head}, p['scan_model'])
        #
        # else:
        #     print('No new lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
        #     print('Lowest loss head is %d' %(best_loss_head))

        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(predictions_val, compute_confusion_matrix=False)
        print(clustering_stats)

        # Checkpoint
        print(f"Checkpoint in {p['scan_checkpoint']}")
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head,
                    'hungarian_match': clustering_stats['hungarian_match']},
                   p['scan_checkpoint'])

    # Evaluate and save the final model
    print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions_val = get_scan_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(predictions_val,
                                          class_names=val_dataset.dataset.classes,
                                          compute_confusion_matrix=True,
                                          confusion_matrix_file=os.path.join(p['scan_dir'], 'confusion_matrix.png'))
    print(clustering_stats)


if __name__ == '__main__':
    fire.Fire(run_scan_trainer)
