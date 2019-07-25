import torch
import numpy as np
import cv2
from PIL import Image,ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import h5py
from skimage import io as skio
import pathlib
import random
import os

class DatasetImageMaskGlobal(Dataset):
    def __init__(self, file_names,object_type, mode):
        self.file_names  = file_names
        self.object_type = object_type
        self.mode = mode
        #print (self.object_type)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
    
        img_file_name = self.file_names[idx]

        with h5py.File(img_file_name, 'r') as data:
            image = data['img'].value
            image = np.pad(image,pad_width=((5,5),(5,5)),mode='constant')
            image = np.expand_dims(image,axis=0)

            mask = data['mask'].value
            mask = np.pad(mask,pad_width=((5,5),(5,5)),mode='constant')
            mask = mask.astype(np.uint8)
            mask = np.expand_dims(mask, 0)


        if self.mode == 'train':    
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long()
        if self.mode == 'valid':
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long()


class DatasetImageMaskLocal(Dataset):

    def __init__(self, file_names,object_type, mode):
        self.file_names  = file_names
        self.object_type = object_type
        self.mode = mode
        #print (self.object_type)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
    
        img_file_name = self.file_names[idx]

        with h5py.File(img_file_name, 'r') as data:
            image = data['img'].value
            image = np.pad(image,pad_width=((5,5),(5,5)),mode='constant')
            image = np.expand_dims(image,axis=0)

            mask = data['mask'].value
            mask = np.pad(mask,pad_width=((5,5),(5,5)),mode='constant')
            mask = mask.astype(np.uint8)
            mask = np.expand_dims(mask, 0)

            coord = data['coord'].value


        if self.mode == 'train':    
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long(),torch.from_numpy(coord)
        if self.mode == 'valid':
            return torch.from_numpy(image).float(),torch.from_numpy(mask).long()

