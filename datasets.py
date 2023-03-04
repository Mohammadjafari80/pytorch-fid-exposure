
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from constants import CIFAR10_PATH, CIFAR100_PATH, MNIST_PATH, FMNIST_PATH, SVHN_PATH, MVTEC_PATH, ADAPTIVE_PATH, mvtec_labels
import torchvision.transforms.functional as F


class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __getitem__(self, index):
        image = self.data[index]
        image = F.to_pil_image(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.data)
    
def get_datasets(normal_dataset:str, normal_class_indx:int, exposure_dataset:str):

    transform = transforms.ToTensor()
    normal_data = get_normal_class(dataset=normal_dataset, normal_class_indx=normal_class_indx, transform=transform)
    exposure_data = get_exposure(dataset=exposure_dataset, normal_dataset=normal_dataset, normal_class_indx=normal_class_indx)

    normal_data = GeneralDataset(data=normal_data, transform=transform)
    exposure_data = GeneralDataset(data=exposure_data, transform=transform)

    return normal_data, exposure_data
    


####################
#  Normal Datastes #
####################

def get_normal_class(dataset='cifar10', normal_class_indx = 0,  transform=None):

    if dataset == 'cifar10':
        return get_CIFAR10_normal(normal_class_indx)
    elif dataset == 'cifar100':
        return get_CIFAR100_normal(normal_class_indx)
    elif dataset == 'mnist':
        return get_MNIST_normal(normal_class_indx)
    elif dataset == 'fashion':
        return get_FASHION_MNIST_normal(normal_class_indx)
    elif dataset == 'svhn':
        return get_SVHN_normal(normal_class_indx)
    elif dataset == 'mvtec':
        return get_MVTEC_normal(normal_class_indx)
    else:
        raise Exception("Dataset is not supported yet. ")


def get_CIFAR10_normal(normal_class_indx:int):
    trainset = CIFAR10(root=CIFAR10_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data]


def get_CIFAR100_normal(normal_class_indx:int):
    trainset = CIFAR100(root=CIFAR100_PATH, train=True, download=True)
    trainset.targets = sparse2coarse(trainset.targets)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data]


def get_MNIST_normal(normal_class_indx:int):
    trainset = MNIST(root=MNIST_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data]


def get_FASHION_MNIST_normal(normal_class_indx:int):
    trainset = FashionMNIST(root=FMNIST_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data]

def get_SVHN_normal(normal_class_indx:int):
    trainset = SVHN(root=SVHN_PATH, split='train', download=True)
    trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx].transpose(0, 2, 3, 1)
    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data]


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, normal=True):
        self.transform = transform
        if train:
            self.data = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          self.data = image_files

        self.data.sort(key=lambda y: y.lower())
        self.data = [Image.open(x).convert('RGB') for x in self.data]
        self.train = train

    def __getitem__(self, index):
        image_file = self.data[index]
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.data)


def get_MVTEC_normal(normal_class_indx):
    normal_class = mvtec_labels[normal_class_indx]
    trainset = MVTecDataset(MVTEC_PATH, normal_class, train=True)
    return  [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data]


######################
#  Exposure Datastes #
######################

def get_exposure(dataset:str='cifar10', normal_dataset:str='cifar100', normal_class_indx:int = 0):
    if dataset == 'cifar10':
        return get_CIFAR10_exposure(normal_dataset, normal_class_indx)
    elif dataset == 'cifar100':
        return get_CIFAR100_exposure(normal_dataset, normal_class_indx)
    elif dataset == 'mnist':
        return get_MNIST_exposure(normal_dataset, normal_class_indx)
    elif dataset == 'fashion':
        return get_FASHION_MNIST_exposure(normal_dataset, normal_class_indx)
    elif dataset == 'svhn':
        return get_SVHN_exposure(normal_dataset, normal_class_indx)
    elif dataset == 'mvtec':
        return get_MVTEC_exposure(normal_dataset, normal_class_indx)
    elif dataset == 'adaptive':
        return get_ADAPTIVE_exposure(normal_dataset, normal_class_indx, count)
    else:
        raise Exception("Dataset is not supported yet. ")
    

def copy_dataset(dataset , target_count:int):
    while target_count > len(dataset):
        dataset = torch.cat((dataset, dataset.data), 0)

    return dataset


def get_CIFAR10_exposure(normal_dataset:str, normal_class_indx:int ):
    exposure_train = CIFAR10(root=CIFAR10_PATH, train=True, download=True)

    if normal_dataset.lower() == 'cifar10':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]

def get_CIFAR100_exposure(normal_dataset:str, normal_class_indx:int ):
    exposure_train = CIFAR100(root=CIFAR100_PATH, train=True, download=True)
    exposure_train.targets = sparse2coarse(exposure_train.targets)

    if normal_dataset.lower() == 'cifar100':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]


def get_MNIST_exposure(normal_dataset:str, normal_class_indx:int):    
    exposure_train = MNIST(root=MNIST_PATH, train=True, download=True)

    if normal_dataset.lower() == 'mnist':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]


def get_FASHION_MNIST_exposure(normal_dataset:str, normal_class_indx:int):    
    exposure_train = FashionMNIST(root=FMNIST_PATH, train=True, download=True)

    if normal_dataset.lower() == 'fmnist':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]


def get_SVHN_exposure(normal_dataset:str, normal_class_indx:int):    
    exposure_train = SVHN(root=SVHN_PATH, split='train', download=True)

    if normal_dataset.lower() == 'svhn':
        exposure_train.data = exposure_train.data[np.array(exposure_train.labels) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)
    del exposure_train

    return [F.to_tensor(np.array(x).astype(np.uint8).transpose(1, 2, 0)) for x  in exposure_data]


def get_ADAPTIVE_exposure(normal_dataset:str, normal_class_indx:int,count:int):
    exposure_data = []
    try:
        exposure_path = glob(os.path.join(ADAPTIVE_PATH, normal_dataset, f'{normal_class_indx}', "*.npy"), recursive=True)
        for path in exposure_path:
            exposure_data += np.load(path).tolist()
    except:
        raise ValueError('Wrong Exposure Address!')
        exit()

    exposure_data = torch.tensor(exposure_data)

    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return exposure_data

class MVTecDatasetExposure(torch.utils.data.Dataset):
    def __init__(self, root, category=None, transform=None):
        self.transform = transform
        self.data = glob(os.path.join(root, "**", "*.png"), recursive=True)

        if category is not None:
          class_files = glob(os.path.join(root, category, "**", "*.png"), recursive=True)
          self.data = list(set(self.data) - set(class_files))

        self.data.sort(key=lambda y: y.lower())
        self.data = np.array([np.array(Image.open(x).convert('RGB')) for x in self.data])


def get_MVTEC_exposure(normal_dataset:str, normal_class_indx:int):    
    exposure_data = torch.tensor(MVTecDatasetExposure(root=MVTEC_PATH).data)

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]



def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]
