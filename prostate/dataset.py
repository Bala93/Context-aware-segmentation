import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import glob
import os 


class TrainData(Dataset):

    def __init__(self, root):

        files = glob.glob(os.path.join(root,'*.h5'))
        self.examples = sorted(files)

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):

        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            input = data['img'].value
            target = data['mask'].value.astype(np.uint8)
            target[target==255] = 1

        return (torch.from_numpy(input), torch.from_numpy(target))


class TrainDataStaticLocal(Dataset):

    def __init__(self, root):

        files = glob.glob(os.path.join(root,'*.h5'))
        self.examples = sorted(files)

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):

        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            input = data['img'].value
            target = data['mask'].value.astype(np.uint8)
            target[target==255] = 1
            coords = data['coord'].value #Center Coordinate of the local ROI(for each input in the batch).

        return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords))   
 

class TrainDataDynamicLocal(Dataset):

    def __init__(self, root):

        files = glob.glob(os.path.join(root,'*.h5'))
        self.examples = sorted(files)

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):

        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            input  = data['img'].value
            target = data['mask'].value.astype(np.uint8)
            target[target==255] = 1
            coords = data['coord'].value #Center Coordinate of the local ROI(for each input in the batch).
            bbox   = data['bbox'].value #ROI Size(bbox x bbox) for the particular batch.

        return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords),bbox)   

class ValidData(Dataset):

    def __init__(self, root):

        files = glob.glob(os.path.join(root,'*.h5'))
        self.examples = sorted(files)

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):

        fname = self.examples[i]
        with h5py.File(fname, 'r') as data:
            input = data['img'].value
            target = data['mask'].value.astype(np.uint8)
            target[target==255] = 1

        return (torch.from_numpy(input), torch.from_numpy(target))

