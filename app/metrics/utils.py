import torch
import torch.nn.functional as F


__all__ = ['iou_mean', 'iou', 'pixel_accuracy']


def iou_mean(predict: torch.Tensor, label: torch.Tensor, n_classes: int):
    iou_m = 0.
    for i in range(n_classes):
       iou_m += iou(predict[i], label[i]).item()
    return iou_m / n_classes


def iou(predict: torch.Tensor, label: torch.Tensor):
    if torch.sum(predict) == 0 and torch.sum(label) == 0:
        return torch.Tensor([1.,])
    overlap = torch.sum(predict * label)
    union = torch.sum(predict) + torch.sum(label) - overlap
    if union == 0.:
        union = 1.
    return overlap / union


def pixel_accuracy(predict: torch.Tensor, label: torch.Tensor):
    correct = torch.eq(predict, label).int()
    accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy