import imp
from turtle import forward
import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
from numpy.linalg import norm
from audtorch.metrics.functional import pearsonr

from torch.autograd import gradcheck

def sparsity(arr, batch_size, lamda2):#arr==[batch_size*32]
    loss =torch.norm(arr)
    return lamda2*loss


def smooth(arr, lamda1):#arr==[batch_size*32]
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return lamda1*loss


class MIL_loss(torch.nn.Module):
    def __init__(self):
        super(MIL_loss, self).__init__()
        self.criterion = torch.nn.BCELoss() #计算目标值和预测值之间的二进制交叉熵损失函数

    def forward(self, score_normal, score_abnormal, nlabel, alabel):
        label = torch.cat((nlabel, alabel), 0) #[2batch_size]
        score = torch.cat((score_normal, score_abnormal), 0) #[2batch_size,1,1]
        score = score.squeeze() #[2batch_size]
        label = label.cuda() #复制到GPU中
        loss_mil = self.criterion(score, label)  # BCE loss in the score space
        return loss_mil


def train(nloader, aloader, model, batch_size, optimizer, device, iteration):
    with torch.set_grad_enabled(True):
        model.train() #开启参数更新

        ninput, nlabel = next(nloader)#iterator的函数[b,10,32,2048] [b,10]
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device) #[2b,10,32,2048]

        score_abnormal, score_normal, scores, score_select_abnormal, score_select_normal, normal_similarity, abnormal_similarity = model(input)

        scores = scores.view(-1, 1)#[2batch_size,32,1]->[2batch_size*32,1]
        scores = scores.squeeze()#[2batch_size*32]

        nlabel = nlabel[0:batch_size] #[batch_size],视频级的标签
        alabel = alabel[0:batch_size] #[batch_size],视频级的标签

        mil = MIL_loss()
        loss_mil = mil(score_normal, score_abnormal, nlabel, alabel)
        loss_sparse = sparsity(scores, batch_size, 1e-2)
        loss_smooth = smooth(scores, 1e-4)
        
        cost = loss_mil + loss_sparse + loss_smooth

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


