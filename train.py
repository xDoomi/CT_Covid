import os
import argparse
import yaml
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from segmentation_models_pytorch import PSPNet
from app.dataset.datasetCT import DatasetCT, DatasetAugmentCT
from app.metrics.dice_score import dice_loss
from app.metrics.utils import iou_mean, pixel_accuracy


def train(rank, world_size, train_ds_all, val_ds, cfg):
    print(f"Running DDP on rank {rank}.")
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    n_channels = cfg['DATASET']['num_channels']
    n_classes = cfg['DATASET']['num_classes']
    batch_size = int(cfg['TRAIN']['batch_size'] / world_size)

    torch.cuda.set_device(rank)
    model = PSPNet(
        encoder_name=cfg['MODEL']['encoder'],
        encoder_weights=None,
        in_channels=n_channels,
        classes=n_classes
    ).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.Adam(ddp_model.parameters())
    criterion = nn.CrossEntropyLoss().cuda(rank)

    train_sample = DistributedSampler(train_ds_all)

    train_loader = DataLoader(train_ds_all, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, sampler=train_sample)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, 
                                pin_memory=True)

    log_interval = len(train_ds_all) // batch_size
    epochs_train_ls, epochs_val_ls = [], []
    epochs_val_cor, epochs_val_iou = [], []
    epochs = cfg['TRAIN']['num_epoch']

    dist.barrier()
    start = datetime.now()
    for epoch in range(epochs):
        train_sample.set_epoch(epoch)
        tr_loss = 0
        val_loss = 0
        pixel_cor = 0
        iou = 0
        ddp_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().cuda(rank, non_blocking=True)
            target = target.float().cuda(rank, non_blocking=True)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, torch.argmax(target, dim=1)) \
                    + dice_loss(F.softmax(output, dim=1),
                                target,
                                multiclass=True)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            if (batch_idx % log_interval == 0 and rank == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data))

        ddp_model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data = data.float().cuda(rank, non_blocking=True)
                target = target.float().cuda(rank, non_blocking=True)
                output = ddp_model(data)
                loss = criterion(output, torch.argmax(target, dim=1)) \
                    + dice_loss(F.softmax(output, dim=1),
                                target,
                                multiclass=True)
                predict = F.softmax(output, dim=1)[0]
                predict = F.one_hot(predict.argmax(dim=0), n_classes).permute(2, 0, 1)
                pixel_cor += pixel_accuracy(predict, target.squeeze())
                iou += iou_mean(predict, target.squeeze(), n_classes)
                val_loss += loss.item()

        if rank == 0:
            print('Train average loss: {:.4f}, Validation average loss: {:.4f}'.format(
                batch_size * cfg['TRAIN']['batch_size'] * tr_loss / len(train_loader.dataset),
                val_loss / len(val_loader.dataset)
            ))
            print('Validation pixel correct: {:.2f}, Validation IoU mean: {:.2f}'.format(
                pixel_cor / len(val_loader.dataset),
                iou / len(val_loader.dataset)
            ))
            print('-----------------------------------------------------')
            epochs_train_ls.append(batch_size * cfg['TRAIN']['batch_size'] * tr_loss / len(train_loader.dataset))
            epochs_val_ls.append(val_loss / len(val_loader.dataset))
            epochs_val_cor.append(pixel_cor / len(val_loader.dataset))
            epochs_val_iou.append(iou / len(val_loader.dataset))

    if rank == 0:
        print("Training complete in: " + str(datetime.now() - start))
        path_save = Path('save')
        torch.save(ddp_model.state_dict(), path_save / 'model')
        print('Model saved')
        with open(path_save / 'results.txt', 'w') as file:
            print(*epochs_train_ls, sep=',', file=file)
            print(*epochs_val_ls, sep=',', file=file)
            print(*epochs_val_cor, sep=',', file=file)
            print(*epochs_val_iou, sep=',', file=file)

    dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_np(images_path, masks_path):
    images = np.load(images_path).astype(np.float32)
    masks = np.load(masks_path).astype(np.int8)
    return images, masks


def main(cfg, n_gpus):
    train_images, train_masks = load_np(cfg['DATASET']['train_images'], 
                                        cfg['DATASET']['train_masks'])
    val_images, val_masks = load_np(cfg['DATASET']['val_images'], 
                                    cfg['DATASET']['val_masks'])

    min_bound, max_bound = cfg['DATASET']['min_bound'], cfg['DATASET']['max_bound']

    train_ds = DatasetCT(np.squeeze(train_images), train_masks, min_bound, 
                                max_bound, n_classes=cfg['DATASET']['num_classes'])

    train_ds_aug = DatasetAugmentCT(np.squeeze(train_images), train_masks, min_bound, 
                                max_bound, n_classes=cfg['DATASET']['num_classes'])
    train_ds_all = train_ds + train_ds_aug

    val_ds = DatasetCT(np.squeeze(val_images), val_masks, min_bound, 
                        max_bound, n_classes=cfg['DATASET']['num_classes'])
    
    mp.spawn(train,
        args=(n_gpus, train_ds_all, val_ds, cfg),
        nprocs=n_gpus,
        join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('-cfg', metavar='FILE', type=str, default='app/configs/unet_resnet.yaml')

    args = parser.parse_args()
    
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg['TRAIN']['seed'])

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 4, f"Requires at least 4 GPUs to run, but got {n_gpus}"

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '29500'
    os.environ["NCCL_DEBUG"] = "INFO"

    main(cfg, n_gpus)