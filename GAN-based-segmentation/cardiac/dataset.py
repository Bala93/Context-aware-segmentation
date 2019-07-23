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
            
            # Padding is done to compensate for the convolution
            input = np.pad(input,((5,5),(5,5)), mode='constant')
            target = np.pad(target,((5,5),(5,5)),mode='constant')

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

            input = np.pad(input,((5,5),(5,5)), mode='constant')
            target = np.pad(target,((5,5),(5,5)),mode='constant')

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
            input = data['img'].value
            input = np.transpose(input,[2,0,1]) #Transform input of dimension HxWxC to CxWxH.  

            target = data['mask'].value.astype(np.uint8)
            target = np.transpose(target,[2,0,1])

            coords = data['coord'].value #Center Coordinate of the local ROI(for each input in the batch).
            bbox   = data['bbox'].value #ROI Size(bbox x bbox) for the particular batch.
            # Create a big canvas and place the image inside, so this can be used to crop the roi with square dimension
            input_canvas  = np.pad(input,pad_width=((0,0),(75,75),(75,75)),mode='constant')
            target_canvas   = np.pad(target,pad_width=((0,0),(75,75),(75,75)),mode='constant')

        return (torch.from_numpy(np.expand_dims(input,1)),torch.from_numpy(np.expand_dims(input_canvas,1)),torch.from_numpy(np.expand_dims(target,1)).long(),torch.from_numpy(np.expand_dims(target_canvas,1)).long(), coords, bbox)   

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

            input = np.pad(input,((5,5),(5,5)), mode='constant')
            target = np.pad(target,((5,5),(5,5)),mode='constant')

        return (torch.from_numpy(input), torch.from_numpy(target))

