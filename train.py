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



def center(arr,lamda3):#arr==[batch_size*32]
    cen = torch.mean(arr)
    loss = (torch.norm(arr-cen, dim=0, p=2)**2)/len(arr)
    return lamda3*loss

def sparsity(arr, batch_size, lamda2):#arr==[batch_size*32],lamda2==8e-3
    loss =torch.norm(arr)
    return lamda2*loss


def smooth(arr, lamda1):#arr==[batch_size*32],lamda1==8e-4
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    #arr  == [1,2,3,4,5]
    #arr2 == [2,3,4,5,5]
    loss = torch.sum((arr2-arr)**2)#/arr.size(0)

    return lamda1*loss

def grad(arr, lamda1):#arr==[batch_size*32],lamda1==8e-4
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    #arr  == [1,2,3,4,5]
    #arr2 == [2,3,4,5,5]
    arr2 = arr2-arr
    idx1 = torch.topk(arr2, 1)[1]#取top3异常幅值,topk[0]返回值,topk[1]返回索引,[self.batch_size,3,1]
    idx2 = torch.topk((-1*arr2), 1)[1]
    loss = (arr2[idx1] + abs(arr2[idx2]))/2
    return loss*lamda1



def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)

#没用上？应该是去掉了
class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))

class MIL_loss(torch.nn.Module):
    def __init__(self):
        super(MIL_loss, self).__init__()
        self.criterion = torch.nn.BCELoss() #计算目标值和预测值之间的二进制交叉熵损失函数

    def forward(self, score_normal, score_abnormal, nlabel, alabel):
        #len = score_abnormal.size()
        #print(len)
        #nlabel = torch.tensor([0.0])
        #nlabel = nlabel.repeat(len)
        #alabel = torch.tensor([1.0])
        #alabel = alabel.repeat(len)
        #label = torch.tensor([nlabel,alabel])
        label = torch.cat((nlabel, alabel), 0) #[2batch_size]
        #score_abnormal = score_abnormal.unsqueeze(0) #[batch_size,1,1]
        #score_normal = score_normal.unsqueeze(0) #[batch_size,1,1]
        #print(score_abnormal)
        #print(score_normal)
        #feat_n==feat_a==[10,self.batch_size,3,2048]

        score = torch.cat((score_normal, score_abnormal), 0) #[2batch_size,1,1]
        score = score.squeeze() #[2batch_size]
        #score = torch.tensor([score_normal,score_abnormal])

        label = label.cuda() #复制到GPU中
        #print(label)
        loss_mil = self.criterion(score, label)  # BCE loss in the score space

        return loss_mil

class MBL_loss(torch.nn.Module):
    def __init__(self):
        super(MBL_loss, self).__init__()
        self.criterion = torch.nn.BCELoss() #计算目标值和预测值之间的二进制交叉熵损失函数

    def forward(self, score_normal, score_abnormal,nlabel,alabel):

        len = score_abnormal.size()
        #print(len)
        nlabel = torch.tensor([0.0])
        nlabel = nlabel.repeat(len)
        alabel = torch.tensor([1.0])
        alabel = alabel.repeat(len)
        label = torch.cat((nlabel, alabel), 0)
        #label = torch.tensor([nlabel,alabel])
        score_abnormal = score_abnormal.unsqueeze(1) #[batch_size,1,1]
        score_normal = score_normal.unsqueeze(1) #[batch_size,1,1]
        #feat_n==feat_a==[10,self.batch_size,3,2048]
        score = torch.cat((score_normal, score_abnormal), 0) #[2batch_size,1,1]
        score = score.squeeze() #[2batch_size]
        #score = torch.tensor([score_normal,score_abnormal])

        label = label.cuda() #复制到GPU中
        #print(label)
        loss_mil = self.criterion(score, label)  # BCE loss in the score space

        return loss_mil

class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha #0.0001
        self.margin = margin #100
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        
    def forward(self, feat_n, feat_a):
        loss_abn = torch.abs(self.margin - torch.mean(feat_a, dim=1))
        #[10*self.batch_size,3,2048]->[10*self.batch_size,2048]->[10*self.batch_size]
        #loss_abn = torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1)
        loss_nor = torch.mean(feat_n, dim=1)
        #[10*self.batch_size,3,2048]->[10*self.batch_size,2048]->[10*self.batch_size]
        loss_rtfm = (torch.norm(loss_abn,p=2) + torch.norm(loss_nor,p=2))#二次方的平均
        #print(self.margin - loss_abn)
        #print(loss_abn)
        #print(loss_nor)
        return self.alpha * loss_rtfm

class Similarity_loss(torch.nn.Module):
    def __init__(self, margin):
        super(Similarity_loss, self).__init__()
        self.relu = torch.nn.ReLU()
        self.margin = margin
    def forward(self, feat_n, feat_a):
        loss_abn = torch.mean(feat_a)
        loss_nor = torch.mean(feat_n)
        loss_similarity = self.relu(self.margin - loss_abn + loss_nor)
        return loss_similarity
class DMIL_loss(torch.nn.Module):
    def __init__(self):
        super(DMIL_loss, self).__init__()
        self.criterion = torch.nn.BCELoss() #计算目标值和预测值之间的二进制交叉熵损失函数
    
    def forward(self, score_select_normal, score_select_abnormal, nlabel, alabel):
        nlabel = nlabel.cuda()
        alabel = alabel.cuda()
        loss_nor = (self.criterion(score_select_normal[0],nlabel)+self.criterion(score_select_normal[1],nlabel)+self.criterion(score_select_normal[2],nlabel))/3
        loss_abn = (self.criterion(score_select_abnormal[0],alabel)+self.criterion(score_select_abnormal[1],alabel)+self.criterion(score_select_abnormal[2],alabel))/3
        loss_dmil = loss_abn + loss_nor
        return loss_dmil


def train(nloader, aloader, model, batch_size, optimizer, device, iteration):
    with torch.set_grad_enabled(True):
        model.train() #开启参数更新


        ninput, nlabel = next(nloader)#iterator的函数[b,10,32,2048] [b,10]
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device) #[2b,10,32,2048]


        score_abnormal, score_normal, feat_select_abn, feat_select_normal, afeat_sim_select, nfeat_sim_select, \
        scores, abnormal_logits, normal_logits, feat_magnitudes, score_select_abnormal, score_select_normal, alogits, logits = model(input)

        #mean_nlogit = (nlogit + mean_logit)/2
        #mean_nlogit = mean_nlogit.squeeze()

        scores = scores.view(-1, 1)#[2batch_size,32,1]->[2batch_size*32,1]
        scores = scores.squeeze()#[2batch_size*32]

        #abn_scores = scores[batch_size * 32:]#[batch_size*32]
        #n_scores = scores[:batch_size * 32]

        #score_select_abnormal = score_select_abnormal.view(3, batch_size, -1)
        #score_select_abnormal = score_select_abnormal.squeeze()#[3,batch_size]
        #score_select_normal = score_select_normal.view(3, batch_size, -1)
        #score_select_normal = score_select_normal.squeeze()

        nlabel = nlabel[0:batch_size] #[batch_size],视频级的标签
        alabel = alabel[0:batch_size] #[batch_size],视频级的标签

        #vim_score = ViM(feat_select_normal,score_select_normal,batch_size)
        rtfm = RTFM_loss(1,1)#0.0001, 100
        loss_rtfm = rtfm(feat_select_normal, feat_select_abn)
        rtfl = RTFM_loss(1,1)
        loss_logit = rtfl(nfeat_sim_select, afeat_sim_select)

        #similarity = Similarity_loss(1)
        #loss_logit = similarity(normal_logits, abnormal_logits)


        #loss_dmil = DMIL_loss()
        mil = MIL_loss()
        loss_mil = mil(score_normal, score_abnormal, nlabel, alabel)

        #mbl = MBL_loss()
        #loss_mbl = mbl(score_batch_normal, score_batch_abnormal, nlabel, alabel)

        loss_sparse = sparsity(scores, batch_size, 1e-4)#8e-3
        loss_smooth = smooth(scores, 1e-2)#8e-4
        print(loss_mil)
        #print(loss_mbl)
        print(loss_sparse)
        print(loss_smooth)
        print(loss_logit)
        #print(loss_logit_smooth)
        print(loss_rtfm)
        #loss_smooth_abn = smooth(abn_scores, 8e-4)#8e-4
        #loss_smooth_abn_logit = smooth(alogit, 8e-4)
        
        #loss_smooth_logit =  8e-3*(torch.norm(alogit.mean(1), p=2))
        #loss_logit =  8e-3*(torch.norm(alogit_select.mean(1), p=2))
        #loss_logit =  (torch.norm(100-alogit_select.mean(1), p=2) + torch.norm(nlogit_select.mean(1), p=2))
        #loss_logit = 0.00001 * ((1000-torch.min(alogit_select)) **2 + torch.max(nlogit_select) ** 2)
        #loss_logit = 0.00001 * torch.mean((1000-torch.min(alogit_select, dim=1)+torch.max(nlogit_select, dim=1))**2)
        #loss_smooth_n = smooth(n_scores, 0.01)#8e-4
        #loss_grad = grad(abn_scores, 1)
        #loss_smooth_n = smooth(n_scores, 8e-3)
        #loss_center = center(n_scores,10)
        #loss_logit_a = torch.abs(100-torch.norm(alogit_select.mean(1), p=2))
        #loss_logit_n = torch.norm(nlogit_select.mean(1), p=2)
        #loss_logit = 0.0001*((loss_logit_a + loss_logit_n))**2
        #print(loss_logit_a)
        #print(loss_logit_n)
        #print(loss_logit)
        #print(loss_rtfm)

        #print(torch.mean(torch.norm(feat_select_abn, dim=2,p=2),dim=1))
        #print(torch.mean(nlogit_select, dim=1))
        #print(torch.mean(nlogit_select, dim=1))

        #print(loss_rtfm)

        #print(loss_center)
        #cost = loss_rtfm(feat_select_normal, feat_select_abn) + loss_mil + loss_sparse + loss_smooth_abn
        #if iteration > 200:
        cost = loss_mil + loss_sparse + loss_smooth #+ loss_rtfm + loss_logit#+ loss_mbl
        #else:
        #    cost = loss_mil + loss_sparse + loss_smooth #+ loss_mbl #+loss_rtfm  #+ loss_logit_sparse + loss_logit_smooth #+loss_rtfm##  # #+ loss_logit #+ loss_smooth_logit+loss_rtfm(feat_select_normal, feat_select_abn) #+ loss_logit

        #cost =  loss_rtfm(feat_select_normal, feat_select_abn) + loss_dmil(score_select_normal,score_select_abnormal,nlabel,alabel) + loss_smooth_abn +loss_sparse + loss_center
        #cost = loss_mil(score_normal, score_abnormal, nlabel, alabel) + loss_smooth_abn + loss_sparse
        
        #cost = loss_rtfm(feat_select_normal, feat_select_abn) + loss_dmil(score_select_normal, score_select_abnormal, nlabel, alabel)+ loss_smooth_abn + loss_sparse
        #cost = loss_rtfm(feat_select_normal, feat_select_abn) + loss_mil(score_normal, score_abnormal, nlabel, alabel) + loss_smooth_abn + loss_sparse + loss_center
        #cost = loss_rtfm(feat_select_normal, feat_select_abn) + loss_dmil(score_select_normal,score_select_abnormal,nlabel,alabel) + loss_smooth_abn +loss_sparse + loss_center
        
        #viz.plot_lines('loss', cost.item())
        #viz.plot_lines('smooth loss abnormal', loss_smooth_abn.item())
        #viz.plot_lines('smooth loss normal', loss_smooth_n.item())
        #viz.plot_lines('sparsity loss', loss_sparse.item())
        #viz.plot_lines('center loss', loss_center.item())
        #print(loss_grad)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        #return mean_logit

