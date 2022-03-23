import torch
import torch.nn.functional as F


__all__ = ['iou_mean', 'pixel_accuracy']


def iou_mean(predict: torch.Tensor, label: torch.Tensor):
    overlap = torch.sum(predict * label, dim=(2, 1))
    union = torch.sum(predict, dim=(2, 1)) + torch.sum(label, dim=(2, 1)) - overlap
    return torch.mean(overlap / union)


def pixel_accuracy(predict: torch.Tensor, label: torch.Tensor):
    with torch.no_grad():
        predict = F.softmax(predict, dim=1)[0].squeeze()
        predict = F.one_hot(predict.argmax(dim=0), 2).permute(2, 0, 1)
        correct = torch.eq(predict, label).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy