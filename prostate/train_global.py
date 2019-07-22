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

from dataset import TrainData,ValidData
from models import UNet,Discriminator


#TODO: Have to look into the logger part 
logging.basicConfig(filename="log_prostrate_global_run6.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M',
                            level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("Run6: S2(LogSoftmax) + C2 with BCE+NLL Loss without Input Masking with batch size 1.")
logger = logging.getLogger(__name__)


def create_datasets(args):

    train_data = TrainData(args.train_path)
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

    model.train()
    running_loss = 0.

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    running_lossG = 0
    running_lossD = 0

    criterionD = nn.BCEWithLogitsLoss()
    criterionG = nn.NLLLoss()

    loss_fake = 0.

    scale = 4e-4

    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(tqdm(data_loader)):

        input, target = data

        input = input.unsqueeze(1).to(args.device)
        target = target.unsqueeze(1).to(args.device)

        input = input.float()
        target = target.float()

        batch_size = input.shape[0]

        outG = modelG(input)

        lossG = criterionG(outG,target.squeeze(1).long())

        for param in modelD.parameters():
            param.requires_grad = True

        optimizerD.zero_grad()

        pred_fake = modelD((torch.exp(outG)).detach())
        fake_label = torch.zeros(pred_fake.shape).to(args.device)
        lossD_fake = criterion_dis(pred_fake, fake_label)

        pred_real = modelD(make_one_hot(target.long()).float())
        real_label = torch.ones(pred_real.shape).to(args.device)
        lossD_real = criterion_dis(pred_real, real_label)

        lossD = (loss_D_real + loss_D_fake) * 0.5 
        lossD.backward()
        optimizerD.step()

        for param in netD.parameters():
            param.requires_grad = False

        optimizerG.zero_grad()
        pred_fake = modelD((torch.exp(outG)).detach())
        lossD_adversarial = criterion_dis(pred_fake, real_label)

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
    #print('modle evluatio')
    losses = []
    start = time.perf_counter()
    #ssim_loss = pytorch_ssim.SSIM()
    # print ("Validation started" )
    # logging.info("Epoch {}".format(epoch))
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input, target, coords, fm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            
            #\print(torch.max(target), torch.max(output))
           # mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            #std = std.unsqueeze(1).unsqueeze(2).to(args.device)
           # target = target * std + mean
            #output = output * std + mean

            #norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)

            # logging.info("input")
            input = input.float()
            # logging.info("target")
            target = target.float()
            # logging.info("model")
            output = model(input)

            # logging.info("loss")
            #loss = F.binary_cross_entropy_with_logits(output[:,1,:,:].unsqueeze(1), target.float())
            loss = F.nll_loss(output,target.squeeze(1).long())
            
            #ssim_out = ssim_loss(target, output)  
            #losses_mse.append(loss)
            losses.append(loss)
            # print ("Iteration {}".format(iter))
            #break
        writer.add_scalar('Dev_Loss_nll', np.mean(losses), epoch)
        #writer.add_scalar('Dev_Loss_ssim', np.mean(losses_ssim), epoch)
       
    return np.mean(losses), time.perf_counter() - start



def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        #print('1')
        image -= image.min()
        #print(image.max())
        image /= image.max()
        #print('2')
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
    #size = 30
    model.eval()
    #class_mults = torch.arange(2).unsqueeze(1).unsqueeze(1).float().to(args.device)
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            input, target, coords, fname = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            output = model(input.float())
            #output_class = output*class_mults
            #output_class = torch.sum(output_class, dim = 1).unsqueeze(1)   
            
            output_numpy = output.detach().cpu().numpy()
            output_mask  = np.argmax(output_numpy,axis=1).astype(float)
            # output_numpy[output_numpy>=0.5] == 1
            # output_numpy[output_numpy<0.5] == 0
            # output_mask = (output_numpy>=0.5).astype(np.float)
            print(np.unique(output_mask),np.max(output_numpy),np.min(output_numpy),output_mask.shape)
            output_final = torch.from_numpy(output_mask).unsqueeze(1)

            #output_class_numpy = output_class.detach().cpu().numpy()
            #output_class_mask  = np.argmax(output_class_numpy,axis=1).astype(float)  
            #output_class_final = torch.from_numpy(output_class_numpy)#.unsqueeze(1)
            # print(torch.max(output_final), torch.min(output_final), torch.unique(output_final))
            #batch_size = input.shape[0]
            #target = target.float()
            '''
            o = []
            t = []
            #print(coords)

            for i in range(input.shape[0]):
                cx, cy = coords[i][0]
                cx = int(cx.item())
                cy = int(cy.item())
                #cx = cx + 5
                #cy = cy + 5

                o.append(output[i,:,cy-size:cy+size,cx-size:cx+size]) 
                t.append(target[i,:,cy-size:cy+size,cx-size:cx+size])

            o_tensor = torch.stack(o).float().to(args.device)
            t_tensor = torch.stack(t).float().to(args.device)
            '''
            save_image(target.float(), 'Target')
            #save_image(output[:,1,:,:].unsqueeze(1), 'Segmentation')
            #print (output.shape)
            save_image(output_final,'Segmentation')
            #save_image(output_class_final,'Segmentation_Class')
            #save_image(torch.abs(target.float() - output[:,1,:,:].unsqueeze(1).float()), 'Error')
            #save_image(t_tensor, 'TargetPatch')
            #save_image(o_tensor[:,1,:,:].unsqueeze(1), 'SegmentationPatch')
            #save_image(torch.abs(t_tensor.float() - o_tensor[:,1,:,:].unsqueeze(1).float()), 'PatchError')
            break


def save_model_step(args, exp_dir, epoch, model, optimizer, disc, optimizerD, iter):
    #rint("in model save  ")
    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'disc': disc.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'exp_dir': exp_dir,
            'iter': iter
        },
        f=exp_dir / 'model_epoch_{}_iter_{}.pt'.format(epoch, iter)
    )
    #print('save', out)
    #if is_new_best:
    #   shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
def save_model(args, exp_dir, epoch, model, optimizer, disc, optimizerD, dev_nll):
    #rint("in model save  ")
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

    netD = Discriminator(n_channels=2).to(args.device)
    optimizerD = optim.SGD(netD.parameters(),lr=5e-3)

    return netD, optimizerD



def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    #args.data_parallel = True
    #print (args.data_parallel)
    #WARNING!!! Check data parallel
    if args.data_parallel:
        #print('sda')
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
    #optimizer = optim.SGD(params, lr=5e-3)
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
    
    # target = Variable(target)
        
    return target


def main(args):

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume: # TODO: Check this block of code for batch size being used from the saved checkpoint 
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

    dev_nll=0
    dev_time=0

    for epoch in range(start_epoch, args.num_epochs+1):
        scheduler.step(epoch)
        train_lossG, train_lossD, train_time = train_epoch(args, epoch, modelG, modelD, train_loader, optimizerG , optimizerD, writer, display_loader, args.exp_dir)
        print ("Epoch {}".format(epoch))
        # logging.info("Train over")
        print ("Validation for epoch :{}".format(epoch))
        dev_nll, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        
        print ("Visualization for epoch :{}".format(epoch))
        visualize(args, epoch, model, display_loader, writer)
        save_model(args, args.exp_dir, epoch, model, optimizer, disc, optimizerD, dev_nll)
        logging.info(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLossG = {train_lossG:.4g} TrainLossD = {train_lossD:.4g} 'f'DevNLL = {dev_nll:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s')
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(decscription='Train setup for GAN based segmentaiton')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting') # TODO: Check whether this is been used 
    
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
