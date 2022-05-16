import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json


def main(args):

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    path_save = Path.cwd() / 'save/'

    with open(path_save / 'results.txt') as f:
        lines = f.readlines()
        epochs_train_ls= list(map(float, lines[0].strip().split(',')))
        epochs_val_ls = list(map(float, lines[1].strip().split(',')))
        epochs_val_cor = list(map(lambda x: float(x) * 100, lines[2].strip().split(',')))
        epochs_val_iou = list(map(lambda x: float(x) * 100, lines[3].strip().split(',')))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
    
    epochs = cfg['TRAIN']['num_epoch']

    axes[0].plot(range(1, epochs + 1), epochs_train_ls, 'r', label='Train loss')
    axes[0].plot(range(1, epochs + 1), epochs_val_ls, 'b', label='Validation loss')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train and Validation loss')

    axes[1].plot(range(1, epochs + 1), epochs_val_cor, color='orange', label='Pixels correct')
    axes[1].plot(range(1, epochs + 1), epochs_val_iou, color='green', label='IoU mean')
    axes[1].grid(True)
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('%')
    axes[1].set_title('Correct and IoU mean')

    plt.savefig(path_save / 'val_plot.png', format='png')
    with open(path_save / 'val_metrics.json', 'w') as outfile:
        json.dump({'pixel_correct' : epochs_val_cor[-1], 'iou': epochs_val_iou[-1]}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate validation script')
    parser.add_argument('-cfg', metavar='FILE', type=str, default='app/configs/fpn_efficientnet.yaml')
    args = parser.parse_args()
    main(args)