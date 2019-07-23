import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import argparse

from dataset import TrainDataStaticLocal,ValidData
from models import UNet,StaticContextDiscriminator

logging.basicConfig(filename="log_prostrate_local.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M',
                            level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("EXP DESCRIPTION")
logger = logging.getLogger(__name__)


def create_datasets(args):

    train_data = TrainDataStaticLocal(args.train_path)
    dev_data = ValidData(args.validation_path)

    return dev_data, train_data


def create_data_loaders(args):

    dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False,
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
    )

    return train_loader, dev_loader, display_loader

def train_epoch(args, epoch, modelG, modelD, data_loader, optimizerG, optimizerD, writer, display_loader, exp_dir):

    modelG.train()
    modelD.train()

    running_loss = 0.

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    size = 25 #ROI Size - 50x50.

    running_lossG = 0
    running_lossD = 0

    criterionD = nn.BCEWithLogitsLoss()
    criterionG = nn.NLLLoss()

    loss_fake = 0.

    scale = 4e-4

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(tqdm(data_loader)):

        input, target, coord = data

        input = input.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).to(args.device)

        input = input.float()
        target = target.float()

        batch_size = input.shape[0]

        outG = modelG(input)
        lossG = criterionG(outG,target.squeeze(1).long())
        
        o = []
        t = []
        
        for i in range(input.shape[0]):
            cx, cy = coord[i][0]
            cx = int(cx.item())
            cy = int(cy.item())
            o.append(torch.exp(outG[i,:,cy-size:cy+size,cx-size:cx+size]))
            t.append((make_one_hot(target[i,:,cy-size:cy+size,cx-size:cx+size].unsqueeze(0).long()).squeeze(0)))
            
        o_tensor = torch.stack(o).to(args.device)
        t_tensor = torch.stack(t).float().to(args.device)

        for param in modelD.parameters():
            param.requires_grad = True

        optimizerD.zero_grad()

        pred_fake = modelD(o_tensor.detach(),(torch.exp(outG)).detach())
        fake_label = torch.zeros(pred_fake.shape).to(args.device)
        lossD_fake = criterionD(pred_fake, fake_label)

        pred_real = modelD(t_tensor.float(),make_one_hot(target.long()).float())
        real_label = torch.ones(pred_real.shape).to(args.device)
        lossD_real = criterionD(pred_real, real_label)

        lossD = (lossD_real + lossD_fake) * 0.5 
        lossD.backward()
        optimizerD.step()

        for param in modelD.parameters():
            param.requires_grad = False

        optimizerG.zero_grad()
        pred_fake = modelD(o_tensor.detach(),(torch.exp(outG)).detach())
        lossD_adversarial = criterionD(pred_fake, real_label)

        lossD_adversarial = (lossD_adversarial)*scale
        lossGan = lossG + lossD_adversarial

        lossGan.backward()      
        optimizerG.step()

        writer.add_scalar('GenLoss', lossG.item(), global_step + iter)
        writer.add_scalar('DiscLoss', lossD.item(), global_step + iter)
        writer.add_scalar('lossD_fake', lossD_fake.item(), global_step+iter)
        writer.add_scalar('lossD_real', lossD_real.item(), global_step+iter)

    return lossG.item(), lossD.item(), time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input, target = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            
            input = input.float()
            target = target.float()
            output = model(input)

            loss = F.nll_loss(output,target.squeeze(1).long())
            losses.append(loss)
        
        writer.add_scalar('Dev_Loss_nll', np.mean(losses), epoch)
       
    return np.mean(losses), time.perf_counter() - start



def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input, target = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            output = model(input.float())
            
            output_numpy = output.detach().cpu().numpy()
            output_mask  = np.argmax(output_numpy,axis=1).astype(float)
            #print(np.unique(output_mask),np.max(output_numpy),np.min(output_numpy),output_mask.shape)
            output_final = torch.from_numpy(output_mask).unsqueeze(1)

            save_image(target.float(), 'Target')
            save_image(output_final,'Segmentation')
            break #Visualize a single batch of images.


def save_model(args, exp_dir, epoch, model, optimizer, disc, optimizerD, dev_nll):
    if epoch%5 == 0:
        out = torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'disc': disc.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'dev_nll':dev_nll
            },
            f=exp_dir / 'model_{}.pt'.format(epoch)
        )



def build_model(args):
    model = UNet(1,2).to(args.device)
    return model

def build_discriminator():

    netD = StaticContextDiscriminator(n_channels=2).to(args.device)
    optimizerD = optim.SGD(netD.parameters(),lr=5e-3)

    return netD, optimizerD



def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    #WARNING!!! Check data parallel
    if args.data_parallel:
       model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    disc, optimizerD = build_discriminator()

    if args.data_parallel:
       disc = torch.nn.DataParallel(disc)    
    disc.load_state_dict(checkpoint['disc'])

    optimizerD.load_state_dict(checkpoint['optimizerD'])
    return checkpoint, model, optimizer, disc, optimizerD


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1) 
        
    return target


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume: 
        print('resuming model, batch_size', args.batch_size)
        checkpoint, model, optimizer, disc, optimizerD = load_model(args.checkpoint)
        bs = args.batch_size
        args = checkpoint['args']
        args.batch_size = bs
        start_epoch = checkpoint['epoch']
        del checkpoint

    else:

        modelG = build_model(args)
        modelD, optimizerD = build_discriminator()

        if args.data_parallel:
           modelG = torch.nn.DataParallel(modelG)
           modelD = torch.nn.DataParallel(modelD)

        optimizerG = build_optim(args, modelG.parameters())
        start_epoch = 0

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs+1):
        scheduler.step(epoch)
        train_lossG, train_lossD, train_time = train_epoch(args, epoch, modelG, modelD, train_loader, optimizerG , optimizerD, writer, display_loader, args.exp_dir)
        print ("Epoch {}".format(epoch))
        print ("Validation for epoch :{}".format(epoch))
        dev_nll, dev_time = evaluate(args, epoch, modelG, dev_loader, writer)
        
        print ("Visualization for epoch :{}".format(epoch))
        visualize(args, epoch, modelG, display_loader, writer)
        save_model(args, args.exp_dir, epoch, modelG, optimizerG,modelD, optimizerD, dev_nll)
        logging.info(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLossG = {train_lossG:.4g} TrainLossD = {train_lossD:.4g} 'f'DevNLL = {dev_nll:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s')
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for GAN based segmentaiton')
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')


    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
