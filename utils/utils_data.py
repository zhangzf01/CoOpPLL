import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from dassl.data.data_manager import DatasetWrapper
from torchvision.transforms.functional import InterpolationMode
import random
from dassl.data.transforms.autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from dassl.data.transforms.randaugment import RandAugment, RandAugment2, RandAugmentFixMatch
from dassl.data.transforms.transforms import INTERPOLATION_MODES,Cutout,GaussianNoise
import numpy as np
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip
)
from dassl.utils import read_image




def symmetric_partialize(train_labels, partial_rate):
    train_labels=torch.tensor(train_labels)
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY




def corrupt_dataset(train_loader,rate):
    ds = train_loader.dataset.data_source
    labels = [ds[i].label for i in range(len(ds))]
    print('corrupting...')
    im_labels = symmetric_partialize(labels, rate)
    for i in range(len(ds)):
        train_loader.dataset.data_source[i]._label = [im_labels[i],labels[i],i]
    return train_loader,im_labels





class Augmented_Dataset_Wrapper_PLCR(DatasetWrapper):
    def __getitem__(self, idx):
        if not self.is_train: 
            return DatasetWrapper.__getitem__(self,idx)

        cfg = self.cfg
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)



        # self-defined augmentation
        normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        input_size = cfg.INPUT.SIZE
        crop_padding = cfg.INPUT.CROP_PADDING
        s_ = cfg.INPUT.RRCROP_SCALE
        n_ = cfg.INPUT.RANDAUGMENT_N
        cutout_n = cfg.INPUT.CUTOUT_N
        cutout_len = cfg.INPUT.CUTOUT_LEN

        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

        transformw1 = Compose([
                Resize(input_size, interpolation=interp_mode),
                RandomCrop(input_size, padding=crop_padding),
                ToTensor(),
                normalize,
            ])
        transformw2 = Compose([
                Resize(input_size, interpolation=interp_mode),
                RandomHorizontalFlip(),
                RandomCrop(input_size, padding=crop_padding),
                RandomResizedCrop(input_size, scale=s_, interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
                normalize,
            ])
        transforms = Compose([
                Resize(input_size, interpolation=interp_mode),
                RandomHorizontalFlip(),
                RandomCrop(input_size, padding=crop_padding),
                RandAugmentFixMatch(n_),
                ToTensor(),
                Cutout(cutout_n, cutout_len),
                normalize,
                GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)
            ])


        output["img"] = [transformw1(img0), transformw2(img0),transforms(img0)]
        return output


class Augmented_Dataset_Wrapper_PiCO(DatasetWrapper):
    def __getitem__(self, idx):
        if not self.is_train: 
            return DatasetWrapper.__getitem__(self,idx)

        cfg = self.cfg
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)


        # self-defined augmentation
        normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        input_size = cfg.INPUT.SIZE

        transformw = Compose([
                RandomResizedCrop(size=input_size, scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                RandomApply([
                    ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                RandomGrayscale(p=0.2),
                ToTensor(), 
                normalize,
            ])
        
        transforms = Compose([
                RandomResizedCrop(size=input_size, scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                RandAugment(3, 5),
                ToTensor(), 
                normalize,
            ])





        output["img"] = [transformw(img0), transforms(img0)]
        return output