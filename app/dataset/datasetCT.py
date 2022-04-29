import numpy as np
import random
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine
from torchvision.transforms.functional import (
    affine,
    rotate, 
    hflip,
    vflip,
    to_tensor,
    get_image_num_channels,
    get_image_size
    )


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
            masks = np.stack((masks[..., 0] + masks[..., 1], masks[..., 3]))
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
            myRandomRotation(),
            myRandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            myRandomAxisFlip()
        ])

    def __getitem__(self, index):
        img = preprocessing.normalize(self.images[index])
        mask = self.masks[index]
        img, mask = self.transforms(img, mask)
        mask = torch.stack((mask[0], torch.logical_not(mask[0], out=torch.Tensor())))
        return img, mask


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

    def __init__(self, min=0, max=180, p=0.5):
        self.min = min
        self.max = max
        self.p = p

    def __call__(self, img, mask):
        if self.p < torch.rand(1):
            return img, mask
        angle = random.randint(self.min, self.max)
        return rotate(img, angle), rotate(mask, angle)


class myRandomAffine(RandomAffine):

    def __init__(self, degrees = 0, translate=None, scale=None, p=0.5):
        super(myRandomAffine, self).__init__(degrees, translate, scale)
        self.p = p

    def forward(self, img, mask):
        if self.p < torch.rand(1):
            return img, mask

        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        return affine(img, *ret, interpolation=self.interpolation, fill=fill), affine(mask, *ret, interpolation=self.interpolation, fill=fill)


class myRandomAxisFlip():

    def __init__(self, axis=None, p=0.5):
        self.axis = axis
        self.p = p
    
    def __call__(self, img, mask):
        if self.p < torch.rand(1):
            return img, mask
        
        if self.axis is None:
            self.axis = random.randrange(0, 2)
        
        if self.axis == 0:
            return hflip(img), hflip(mask)
        elif self.axis == 1:
            return vflip(img), vflip(mask)