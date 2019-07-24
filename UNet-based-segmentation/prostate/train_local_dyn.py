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
from skimage import measure

from utils import visualize,evaluate
from losses import LossMulti, FocalLoss 
from models import UNet
from dataset import DatasetImageMaskLocal

if __name__ == "__main__":

    train_path  =  '/media/htic/NewVolume3/Balamurali/promise_prostate_dataset/train/*.h5'
    val_path = '/media/htic/NewVolume3/Balamurali/promise_prostate_dataset/test/*.h5'
    object_type = 'prostate'
    model_type = 'UNet'
    save_path = '/media/htic/NewVolume5/midl_experiments/nll/{}_{}/models_local_dyn'.format(object_type,model_type)

    use_pretrained = False
    pretrained_model_path = '/media/htic/NewVolume5/midl_experiments/nll/prostate_unet/models_run3/40.pt'
   
    #TODO:Add hyperparams to ArgParse. 
    batch_size = 1
    val_batch_size = 9 
    no_of_epochs = 150

    cuda_no = 0
    CUDA_SELECT = "cuda:{}".format(cuda_no)

    writer = SummaryWriter(log_dir='/media/htic/NewVolume5/midl_experiments/nll/{}_{}/models_local_dyn/summary'.format(object_type,model_type))

    logging.basicConfig(filename="log_{}_run_local_dyn.txt".format(object_type),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M',
                            level=logging.INFO)

    logging.info('Model: UNet + Loss: DynamicContextLoss {}'.format(object_type)) 

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
        print("Loading Model {}".format(os.path.basename(pretrained_model_path)))
        model.load_state_dict(torch.load(pretrained_model_path))
        epoch_start = os.path.basename(pretrained_model_path).split('.')[0]
        print(epoch_start)

    
    trainLoader   = DataLoader(DatasetImageMaskLocal(train_file_names,object_type,mode='train'),batch_size=batch_size)
    devLoader     = DataLoader(DatasetImageMaskLocal(val_file_names,object_type,mode='valid'))
    displayLoader = DataLoader(DatasetImageMaskLocal(val_file_names,object_type,mode='valid'),batch_size=val_batch_size)

    optimizer = Adam(model.parameters(), lr=1e-4)

    criterion_global = LossMulti(num_classes=2,jaccard_weight=0)
    criterion_local = LossMulti(num_classes=2,jaccard_weight=0)

    for epoch in tqdm(range(int(epoch_start)+1,int(epoch_start)+1+no_of_epochs)):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0
        running_loss_local = 0.0

        for i,(inputs,targets,coord) in enumerate(tqdm(trainLoader)):

            model.train()
            inputs   = inputs.to(device)
            targets  = targets.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs) 
                loss_global = criterion_global(outputs,targets)

                local_outputs_ = []
                local_targets_ = []
                for i,coord in enumerate(coord):
                       
                    cx,cy = int(coord[:,0].item()),int(coord[:,1].item())
                    targets_numpy = targets[i].detach().cpu().numpy()
                    region_props = measure.regionprops(targets_numpy.astype('int'))
                    area = region_props[0].area
                    bbox = region_props[0].bbox
                    local_outputs = outputs[i,:,bbox[0]:bbox[2],bbox[1]:bbox[3]]
                    local_targets = targets[i,:,bbox[0]:bbox[2],bbox[1]:bbox[3]]

                    local_outputs_.append(local_outputs)
                    local_targets_.append(local_targets)         
                loss_local = criterion_local(torch.stack(local_outputs_),torch.stack(local_targets_))

                loss = loss_global + loss_local
                writer.add_scalar('loss_global', loss_global, epoch)
                writer.add_scalar('loss_local', loss_local, epoch)
                writer.add_scalar('loss_total', loss.item(), epoch)
                    
                loss.backward()
                optimizer.step()

            running_loss += loss.item()*inputs.size(0)
            running_loss_local += loss_local*inputs.size(0)

        epoch_loss = running_loss / len(train_file_names)
        epoch_loss_local = running_loss_local / len(train_file_names)

        if epoch%1 == 0:
            dev_loss,dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar('loss_valid', dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, val_batch_size)
            print("Global Loss:{} Local Loss:{} Dev Loss:{}".format(epoch_loss,epoch_loss_local,dev_loss))
        else:
            print("Global Loss:{} Local Loss:{}".format(epoch_loss,epoch_loss_local))
        
        logging.info('epoch:{} train_loss:{} train_loss_local:{}'.format(epoch,epoch_loss,epoch_loss_local))
        #if epoch%5 == 0:
        #    torch.save(model.state_dict(),os.path.join(save_path,str(epoch)+'.pt'))
