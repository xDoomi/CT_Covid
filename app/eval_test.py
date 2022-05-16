import argparse
import yaml
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from segmentation_models_pytorch import PSPNet
from dataset.datasetCT import DatasetCT
from metrics.utils import iou, pixel_accuracy
from metrics.dice_score import dice_coeff


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
    min_bound = cfg['DATASET']['min_bound']
    max_bound = cfg['DATASET']['max_bound']

    model_ddp = PSPNet(
        encoder_name=cfg['MODEL']['encoder'],
        encoder_weights=None,
        in_channels=n_channels,
        classes=n_classes
    ).to(device)
    model_ddp = DataParallel(model_ddp)
    model_ddp.load_state_dict(torch.load(path_save / 'model', map_location=device))

    test_images, test_masks = load_np(cfg['DATASET']['test_images'], 
                                        cfg['DATASET']['test_masks'])
    test_ds = DatasetCT(test_images, test_masks, min_bound, max_bound, test=True)
    iou_metrics = 0
    pixel_correct = 0
    dice = 0

    for i in range(len(test_ds)):
        temp = test_ds[i]
        img, target = temp[0], temp[1]
        mask = predict(model_ddp, img, device, n_classes)
        conc_mask = mask[0]
        pixel_correct += pixel_accuracy(conc_mask, target.squeeze())
        iou_metrics += iou(conc_mask, target.squeeze()).item()
        dice += dice_coeff(conc_mask, target.squeeze().to(torch.int64)).item()

    with open(path_save / 'test_metrics.json', 'w') as outfile:
        json.dump({'pixel_correct' : pixel_correct / len(test_ds), 
                    'iou' : iou_metrics / len(test_ds),
                    'dice' : dice / len(test_ds)}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script')
    parser.add_argument('-cfg', metavar='FILE', type=str, default='app/configs/pspnet_efficientnet.yaml')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    main(args, device)