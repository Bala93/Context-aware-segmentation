import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import functools
from collections import namedtuple
from torchvision import models

'''
    Ordinary UNet Conv Block
'''
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation


        init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out
'''
    Ordinary UNet-Up Conv Block
'''
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        init.xavier_uniform_(self.up.weight, gain = np.sqrt(2.0))
        init.constant_(self.up.bias,0)
        init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)

        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out


class UNet(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 1):
        super(UNet, self).__init__()
        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)
        
        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, n_classes, 1, stride=1)


    def forward(self, x):

        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(up1, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)

        return F.log_softmax(self.last(up4), dim=1)        

       
class Discriminator(nn.Module):
    def __init__(self,n_channels=1):
        super(Discriminator,self).__init__()

        self.conv1 = nn.Conv2d(n_channels,32,(9,9), 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,(5,5))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,(5,5))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9216,512) 
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,1)        
        
    def forward(self,x):
        x = F.avg_pool2d(F.leaky_relu(self.conv1(x), 0.2),(2,2))
        x = F.avg_pool2d(F.leaky_relu(self.conv2(x), 0.2),(2,2))
        x = F.avg_pool2d(F.leaky_relu(self.conv3(x), 0.2),(2,2))
        x = x.view(-1,self.num_of_flat_features(x))
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        
        x = self.fc3(x)
        
        return x

    def num_of_flat_features(self,x):
        size=x.size()[1:] 
        num_features=1
        for s in size:
            num_features*=s
        return num_features

        
class GlobalDiscriminator(nn.Module):

    def __init__(self,n_channels=1):

        super(GlobalDiscriminator,self).__init__()
        self.conv1 = nn.Conv2d(n_channels,32,(9,9), 1)
        self.conv2 = nn.Conv2d(32,64,(5,5))
        self.conv3 = nn.Conv2d(64,64,(5,5))
        self.fc1 = nn.Linear(9216,512) 
        self.fc2 = nn.Linear(512,64)

    def forward(self,x):

        x = F.avg_pool2d(F.leaky_relu(self.conv1(x), 0.2),(2,2))
        x = F.avg_pool2d(F.leaky_relu(self.conv2(x), 0.2),(2,2))
        x = F.avg_pool2d(F.leaky_relu(self.conv3(x), 0.2),(2,2))
        x = x.view(-1,self.num_of_flat_features(x))
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)

        return x

    def num_of_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features


class StaticLocalDiscriminator(nn.Module):
    def __init__(self,n_channels=1):
        super(StaticLocalDiscriminator,self).__init__()

        self.conv1 = nn.Conv2d(n_channels,32,(9,9), 1)
        self.conv2 = nn.Conv2d(32,64,(5,5))
        self.conv3 = nn.Conv2d(64,64,(5,5))
        self.fc1 = nn.Linear(576,512) 
        self.fc2 = nn.Linear(512,64)
        
    def forward(self,x):
        x = F.avg_pool2d(F.leaky_relu(self.conv1(x), 0.2),(2,2))
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = F.leaky_relu(self.conv3(x),0.2)
        x = self.fc2(self.fc1(x)) 
        x = x.view(-1,self.num_of_flat_features(x))

        return x

    def num_of_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features


class DynamicLocalDiscriminator(nn.Module):
    def __init__(self,n_channels=1):
        super(DynamicLocalDiscriminator,self).__init__()

        self.conv1 = nn.Conv2d(n_channels,32,(9,9), 1)
        self.conv2 = nn.Conv2d(32,64,(5,5))
        self.conv3 = nn.Conv2d(64,64,(5,5))
        self.gap1 = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self,x):
        x = F.avg_pool2d(F.leaky_relu(self.conv1(x), 0.2),(2,2))
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = F.leaky_relu(self.conv3(x),0.2)
        x = self.gap1(x)
        x = x.view(-1,self.num_of_flat_features(x))

        return x

    def num_of_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features

class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class StaticContextDiscriminator(nn.Module):
    def __init__(self,n_channels=1):
        super(StaticContextDiscriminator, self).__init__()
        self.model_ld = StaticLocalDiscriminator(n_channels)
        self.model_gd = GlobalDiscriminator(n_channels)

        self.concat1 = Concatenate(dim=-1)
        self.linear1 = nn.Linear(128, 1)

    def forward(self, x_local, x_global):
        x_ld = self.model_ld(x_local)
        x_gd = self.model_gd(x_global)
        x = self.linear1(self.concat1([x_ld, x_gd]))
        return x

class DynamicContextDiscriminator(nn.Module):
    def __init__(self,n_channels=1):
        super(DynamicContextDiscriminator, self).__init__()

        self.model_ld = DynamicLocalDiscriminator(n_channels)
        self.model_gd = GlobalDiscriminator(n_channels)

        self.concat1 = Concatenate(dim=-1)
        self.linear1 = nn.Linear(128, 1)

    def forward(self, x_local, x_global):
        x_ld = self.model_ld(x_local)
        x_gd = self.model_gd(x_global)
        x = self.linear1(self.concat1([x_ld, x_gd]))
        return x




