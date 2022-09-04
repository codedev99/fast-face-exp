import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import transforms, datasets
import os
import numpy as np
import collections
import cv2

def loader(path):
    image = np.asarray(cv2.imread(path)).astype(np.uint8)[..., ::-1]    #[BGR --> RGB]
    # image = np.transpose(image, (2,0,1))    # [HWC --> CHW]
    return image.copy() # [H x W x C ]

class InferenceRAM(Dataset):
    def __init__(self, root, loader, transform=None):
        self.root = root
        self.samples = os.listdir(root)
        self.loader = loader
        self.new_samples = self._get_ram_data()
        self.transform = transform
    
    def _get_ram_data(self):
        new_samples = []
        for sample in self.samples:
            path = os.path.join(self.root, sample)
            img = self.loader(path)
            new_samples.append(img)
        
        return new_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.new_samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class FERPlusRam(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.dataset = datasets.DatasetFolder(root, loader=loader, extensions=('.png'))
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples
        self.targets = self.dataset.targets
        self.loader = loader
        self.new_samples = self._get_ram_data()
        self.transform = transform
        self.target_transform = target_transform
    
    def _get_ram_data(self):
        new_samples = []
        for sample in self.samples:
            path = sample[0]
            img = self.loader(path)
            new_samples.append((img, sample[1]))
        
        return new_samples
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample, target = self.new_samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class BalancedSampler(Sampler):
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.targets = dataset.targets
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        self.num_instances, self.classwise_idx = self._get_constraints()
        self.desired_perclass = max(self.num_instances)

    def _get_constraints(self):
        num_instances = [0] * self.num_classes
        classwise_idx = collections.defaultdict(list)
        for i in range(len(self.targets)):
            target = self.targets[i]
            num_instances[target] += 1
            classwise_idx[target].append(i)
        
        return num_instances, classwise_idx
    
    def __len__(self):
        return self.desired_perclass * self.num_classes

    def gen_samples(self):
        stack = []
        for i in range(self.num_classes):
            instance_idx = torch.Tensor(self.classwise_idx[i])
            l = len(instance_idx)
            if self.shuffle:
                instance_idx = instance_idx[torch.randperm(l)]
            desired_idx = instance_idx.repeat(self.desired_perclass // l)
            stack += [desired_idx, instance_idx[:self.desired_perclass % l]]
        stack = torch.cat(stack, dim=0).type(torch.IntTensor)
        if self.shuffle:
            l = len(stack)
            stack = stack[torch.randperm(l)]

        return stack

    def __iter__(self):
        return iter(self.gen_samples())