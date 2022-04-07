import numpy as np
import random
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import rotate, to_tensor


__all__ = ['DatasetCT, DatasetAugmentCT']


class DatasetCT(Dataset):
    
    def __init__(self,
                images: np.ndarray,
                masks: np.ndarray,
                concatenate_class: bool = False,
                test: bool = False):
        super().__init__()

        self.length = len(images)
        self.images = images
        self.concatenate_class = concatenate_class

        if self.concatenate_class:
            masks = np.stack((masks[..., 0] + masks[..., 1], masks[..., 2], masks[..., 3]))
            self.masks = np.transpose(masks, (1, 0, 2, 3))
        elif test:
            self.masks = masks
        else:
            self.masks = np.transpose(masks, (0, 3, 1, 2))

        self.transforms = Compose([
            ToTensor()
        ])
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_norm = preprocessing.normalize(self.images[index])
        mask = self.masks[index]
        return (self.transforms(image_norm), torch.from_numpy(mask))


class DatasetAugmentCT(DatasetCT):

    def __init__(self,
            images: np.ndarray,
            masks: np.ndarray,
            concatenate_class: bool = False):
        super(DatasetAugmentCT, self).__init__(images, masks, concatenate_class)

        self.transforms = myCompose([
            myToTensor(),
            myRandomRotation()
        ])

    def __getitem__(self, index):
        image_norm = preprocessing.normalize(self.images[index])
        mask = self.masks[index]
        return self.transforms(image_norm, mask)


class myCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
    

class myToTensor():
    def __call__(sefl, img, mask):
        return to_tensor(img), torch.from_numpy(mask)


class myRandomRotation():
    def __init__(self):
        self.angle = [90., 180., 270.]

    def __call__(self, img, mask):
        angle = self.angle[random.randrange(0, 3)]
        return rotate(img, angle), rotate(mask, angle)