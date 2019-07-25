import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1,device=None):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(self.device)
        else:
            nll_weight = None
       
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets): 
        
        targets = targets.squeeze(1) 
        loss = (1-self.jaccard_weight) * self.nll_loss(outputs,targets) 
        if self.jaccard_weight:
            eps = 1e-7
            for cls in range(self.num_classes): 
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp() 
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight 

        return loss

class FocalLoss:
    def __init__(self,jaccard_weight=0, class_weights=None,num_classes=1,device=None):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(device)
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.num_classes = num_classes
        self.device = device
    
    def __call__(self,outputs,targets,alpha=4.0,gamma=2.0,eps=1e-7):
        targets = targets.squeeze(1)
        target_one_hot = torch.eye(self.num_classes,device=self.device)[targets]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        cls_prob = outputs.exp() + eps #Counter LogSoftmax with exp to get softmax.
        ce = target_one_hot * -1 * torch.log(cls_prob)
        weight = target_one_hot * ((1-cls_prob) ** gamma)
        fl = alpha * weight * ce
        reduced_fl,_ = torch.max(fl,dim=1)
        loss = torch.mean(reduced_fl)

        return loss
