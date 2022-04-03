import torch
import torch.nn.functional as F


__all__ = ['iou_mean', 'iou', 'pixel_accuracy']


def iou_mean(predict: torch.Tensor, label: torch.Tensor):
    overlap = torch.sum(predict * label, dim=(2, 1))
    union = torch.sum(predict, dim=(2, 1)) + torch.sum(label, dim=(2, 1)) - overlap
    return torch.mean(overlap / union)


def iou(predict: torch.Tensor, label: torch.Tensor):
    overlap = torch.sum(predict * label)
    union = torch.sum(predict) + torch.sum(label) - overlap
    if union == 0.:
        union = 1.
    return overlap / union


def pixel_accuracy(predict: torch.Tensor, label: torch.Tensor):
    correct = torch.eq(predict, label).int()
    accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy