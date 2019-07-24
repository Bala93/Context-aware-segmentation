import torch
import numpy as np
import cv2
from PIL import Image,ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import h5py
from skimage import io as skio
import pathlib
import random
import os

class DatasetImageMaskGlobal(Dataset):
    def __init__(self,file_names,object_type, mode):
        self.file_names  = file_names
        self.object_type = object_type
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
    
        img_file_name = self.file_names[idx]

        with h5py.File(img_file_name, 'r') as data:
            image = data['img'].value
            mask = data['mask'].value

            image = np.expand_dims(image,axis=0)
            mask[mask == 255.] = 1
            mask = mask.astype(np.uint8)
            mask = np.expand_dims(mask, 0)

        if self.mode == 'train':    
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long()

        if self.mode == 'valid':
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long()

class DatasetImageMaskStaticLocal(Dataset):
    def __init__(self, file_names,object_type, mode):
        self.file_names  = file_names
        self.object_type = object_type
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
    
        img_file_name = self.file_names[idx]

        with h5py.File(img_file_name, 'r') as data:
            image = data['img'].value

            mask  = data['mask'].value
            mask[mask == 255.] = 1
            mask = mask.astype(np.uint8)

            coord = data['coord'].value
            
            # Expanding for (hxw)
            image = np.expand_dims(image,axis=0)
            mask  = np.expand_dims(mask, 0)

        if self.mode == 'train':    
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long(), torch.from_numpy(coord)
        if self.mode == 'valid':
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long()


class DatasetImageMaskDynamicLocal(Dataset):

    def __init__(self, file_names,object_type, mode):

        self.file_names  = file_names
        self.object_type = object_type
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
    
        img_file_name = self.file_names[idx]

        with h5py.File(img_file_name, 'r') as data:

            image = data['img'].value
            mask = data['mask'].value
 
            # Transpose to convert from image to tensor dimension 
            image = np.transpose(image,[2,0,1])
            mask = np.transpose(mask,[2,0,1])

            mask[mask == 255.] = 1
            mask = mask.astype(np.uint8)

            coord = data['coord'].value
            bbox  = data['bbox'].value

            # Expanding for (batch x height x width)
            image = np.expand_dims(image,axis=1)
            mask  = np.expand_dims(mask, 1)


        if self.mode == 'train':    
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long(), torch.from_numpy(coord),bbox
        if self.mode == 'valid':
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long(), torch.from_numpy(coord)

