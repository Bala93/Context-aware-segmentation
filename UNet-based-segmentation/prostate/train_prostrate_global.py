import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import numpy as np
import h5py
import torchvision
import random
from tensorboardX import SummaryWriter
from scipy.ndimage.morphology import distance_transform_edt

from utils import visualize,evaluate
from losses import LossMulti 
from models import UNet
from dataset import DatasetImageMaskGlobal

if __name__ == "__main__":

    train_path  =  '/media/htic/NewVolume3/Balamurali/promise_prostate_dataset/train/*.h5'
    val_path = '/media/htic/NewVolume3/Balamurali/promise_prostate_dataset/test/*.h5'
    object_type = 'prostrate'
    model_type = 'UNet'
    save_path = '/media/htic/NewVolume5/midl_experiments/nll/{}_{}/models_run4'.format(object_type,model_type)
    load_path = '/media/htic/NewVolume5/midl_experiments/nll/prostrate_unet/models_run3/40.pt'

    use_pretrained = False
    batch_size = 16
    val_batch_size = 9 
    no_of_epochs = 150

    cuda_no = 0
    CUDA_SELECT = "cuda:{}".format(cuda_no)



    writer = SummaryWriter(log_dir='/media/htic/NewVolume5/midl_experiments/nll/{}_{}/models_run4/summary'.format(object_type,model_type))

    logging.basicConfig(filename="log_{}_run4.txt".format(object_type),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M',
                            level=logging.INFO)

    logging.info('Model: UNet + Loss: FocalLoss(alpha=4) {}'.format(object_type)) 

    train_file_names = glob.glob(train_path)
    random.shuffle(train_file_names)

    val_file_names = glob.glob(val_path)

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=2,input_channels=1)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)

    model = model.to(device)

    # To handle epoch start number and pretrained weight 
    epoch_start = '0'
    if(use_pretrained):
        print("Loading Model {}".format(os.path.basename(load_path)))
        model.load_state_dict(torch.load(load_path))
        epoch_start = os.path.basename(load_path).split('.')[0]
        print(epoch_start)

    
    trainLoader   = DataLoader(DatasetImageMaskGlobal(train_file_names,object_type,mode='train'),batch_size=batch_size)
    devLoader     = DataLoader(DatasetImageMaskGlobal(val_file_names,object_type,mode='valid'))
    displayLoader = DataLoader(DatasetImageMaskGlobal(val_file_names,object_type,mode='valid'),batch_size=val_batch_size)

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss2(num_classes=2,device=device)


    for epoch in tqdm(range(int(epoch_start)+1,int(epoch_start)+1+no_of_epochs)):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        for i,(img_file_name,inputs,targets,_,_) in enumerate(tqdm(trainLoader)):

            model.train()
            inputs   = inputs.to(device)
            targets  = targets.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs) 
                loss_global = criterion_global(outputs,targets,gamma=2,alpha=4)

                writer.add_scalar('loss', loss, epoch)

                loss.backward()
                optimizer.step()

            running_loss += loss.item()*inputs.size(0)

        epoch_loss = running_loss / len(train_file_names)

        if epoch%1 == 0:
            dev_loss,dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar('loss_valid', dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, val_batch_size)
            print("Global Loss:{} Val Loss:{}".format(epoch_loss,dev_loss))
        else:
            print("Global Loss:{} ".format(epoch_loss))
        
        logging.info('epoch:{} train_loss:{} '.format(epoch,epoch_loss))
        if epoch%5 == 0:
            torch.save(model.state_dict(),os.path.join(save_path,str(epoch)+'.pt'))
