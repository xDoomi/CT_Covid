import argparse
import yaml
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from models.simple_unet import UNet
from dataset.datasetCT import DatasetCT
from metrics.utils import iou, pixel_accuracy


def load_np(images_path, masks_path):
    images = np.load(images_path).astype(np.float32)
    masks = np.load(masks_path).astype(np.int8)
    return images, masks


def predict(model,
            img,
            device,
            n_classes):
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    mask = F.softmax(output, dim=1)[0]
    mask = F.one_hot(mask.argmax(dim=0), n_classes).permute(2, 0, 1)
    return mask.cpu()
    

def main(args, device):

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    path_save = Path.cwd() / 'save'

    n_channels = cfg['DATASET']['num_channels']
    n_classes = cfg['DATASET']['num_classes']

    model_ddp = UNet(n_channels, n_classes).to(device)
    #model_ddp = DataParallel(model)
    model_ddp.load_state_dict(torch.load(path_save / cfg['MODEL']['name'], map_location=device))

    test_images, test_masks = load_np(cfg['DATASET']['test_images'], 
                                        cfg['DATASET']['test_masks'])
    test_ds = DatasetCT(test_images, test_masks)
    iou_metrics = 0
    pixel_correct = 0

    for i in range(len(test_ds)):
        temp = test_ds[i]
        img, target = temp[0], temp[1]
        mask = predict(model_ddp, img, device, n_classes)
        conc_mask = mask[0] * mask[1] 
        pixel_correct += pixel_accuracy(conc_mask, target.squeeze())
        iou_metrics += iou(conc_mask, target.squeeze()).item()

    with open(path_save / 'test_metrics.json', 'w') as outfile:
        json.dump({'pixel_correct' : pixel_correct / len(test_ds), 'iou': iou_metrics / len(test_ds)}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script')
    parser.add_argument('-cfg', metavar='FILE', type=str, default='app/configs/simple_unet.yaml')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    main(args, device)