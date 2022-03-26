import argparse
import yaml
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from app.models.simple_unet import UNet
from app.dataset.datasetCT import DatasetCT
from app.metrics.dice_score import dice_loss
from app.metrics.utils import iou_mean, pixel_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_np(images_path, masks_path):
    images = np.load(images_path).astype(np.float32)
    masks = np.load(masks_path).astype(np.int8)
    return images, masks


def train(model, train_loader, val_loader, criterion, optimizer, cfg):
    model.train()
    epochs_train_ls, epochs_val_ls = [], []
    epochs_val_cor, epochs_val_iou = [], []
    epochs = cfg['TRAIN']['num_epoch']
    start = datetime.now()
    for epoch in range(epochs):
        tr_loss = 0
        val_loss = 0
        pixel_cor = 0
        iou = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.argmax(target, dim=1)) \
                    + dice_loss(F.softmax(output, dim=1),
                                target,
                                multiclass=True)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            if (batch_idx % 200 == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data))

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                output = model(data)
                loss = criterion(output, torch.argmax(target, dim=1)) \
                    + dice_loss(F.softmax(output, dim=1),
                                target,
                                multiclass=True)
                predict = F.softmax(output, dim=1)[0]
                predict = F.one_hot(predict.argmax(dim=0), cfg['DATASET']['num_classes']).permute(2, 0, 1)
                pixel_cor += pixel_accuracy(predict, target.squeeze())
                iou += iou_mean(predict, target.squeeze()).item()
                val_loss += loss.item()

        print('Train average loss: {:.4f}, Validation average loss: {:.4f}'.format(
            cfg['TRAIN']['batch_size'] * tr_loss / len(train_loader.dataset),
            val_loss / len(val_loader.dataset)
        ))
        print('Validation pixel correct: {:.2f}, Validation IoU mean: {:.2f}'.format(
            pixel_cor / len(val_loader.dataset),
            iou / len(val_loader.dataset)
        ))
        print('-----------------------------------------------------')
        epochs_train_ls.append(cfg['TRAIN']['batch_size'] * tr_loss / len(train_loader.dataset))
        epochs_val_ls.append(val_loss / len(val_loader.dataset))
        epochs_val_cor.append(pixel_cor / len(val_loader.dataset))
        epochs_val_iou.append(iou / len(val_loader.dataset))

    print("Training complete in: " + str(datetime.now() - start))
    path_save = Path('save')
    torch.save(model.state_dict(), path_save / '{}'.format(cfg['MODEL']['name']))
    print('Model saved: {}'.format(cfg['MODEL']['name']))
    with open(path_save / 'results.txt', 'w') as file:
        print(*epochs_train_ls, sep=',', file=file)
        print(*epochs_val_ls, sep=',', file=file)
        print(*epochs_val_cor, sep=',', file=file)
        print(*epochs_val_iou, sep=',', file=file)


def main(cfg):
    n_channels = cfg['DATASET']['num_channels']
    n_classes = cfg['DATASET']['num_classes']
    model = UNet(n_channels, n_classes).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_images, train_masks = load_np(cfg['DATASET']['train_images'], 
                                        cfg['DATASET']['train_masks'])
    val_images, val_masks = load_np(cfg['DATASET']['val_images'], 
                                    cfg['DATASET']['val_masks'])

    train_ds = DatasetCT(train_images[:30], train_masks[:40])
    val_ds = DatasetCT(val_images, val_masks)
    
    train_loader = DataLoader(train_ds, batch_size=cfg['TRAIN']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds)

    train(model, train_loader, val_loader, criterion, optimizer, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('-cfg', metavar='FILE', type=str, default='app/configs/simple_unet.yaml')

    args = parser.parse_args()
    
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg['TRAIN']['seed'])

    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    main(cfg)
