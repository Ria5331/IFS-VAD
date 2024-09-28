from threading import local
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
import math
from torch import Tensor
from torch.nn import init
from timm.models.layers import DropPath, trunc_normal_
import numpy
import sys

class CycleFC(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)

class CycleMixer(nn.Module):
    def __init__(self, input, hidden, global_stride, local_stride):
        super().__init__()

        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(input)
        self.local_mlp_block1 = CycleFC(input, hidden, (local_stride, 1), 1, 0)
        self.local_mlp_block2 = CycleFC(input, hidden, (global_stride, 1), 1, 0)
        self.local_mlp_block3 = CycleFC(hidden, input, (local_stride, 1), 1, 0)
        self.local_mlp_block4 = CycleFC(hidden, input, (global_stride, 1), 1, 0)

    def forward(self, x):
        bs, t, f = x.size()

        local_f = x
        local_f = self.ln(local_f)
        local_f = (local_f.permute(0, 2, 1)).reshape(bs, f, t, 1)
        local_f1 = self.local_mlp_block1(local_f)
        local_f2 = self.local_mlp_block2(local_f)
        local_f = local_f1 + local_f2
        local_f = self.gelu(local_f)
        local_f1 = self.local_mlp_block3(local_f)
        local_f2 = self.local_mlp_block4(local_f)
        local_f = local_f1 + local_f2
        local_f = (local_f.reshape(bs, f, t)).permute(0, 2, 1)
        local_f = local_f * x + x
        local_f = self.ln(local_f)

        return local_f

class MlpModel(nn.Module):
    def __init__(self,num_blocks,batch_size,n_features):
        super().__init__()
        self.num_blocks = num_blocks
        self.mlp_dim = n_features
        self.batch_size = batch_size
        self.k_abn = 5
        self.k_nor = self.k_abn
        self.mlp_blocks = CycleMixer(self.mlp_dim, 2 * self.mlp_dim,32, 4)
        self.fc1 = nn.Linear(self.mlp_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.w = nn.Parameter(torch.ones(2))
        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,inputs,test=False):

        bs, ncrops, t, f = inputs.size()
        inputs = inputs.reshape(-1, t, f) #[10bs,T,2048]

        features = inputs
        features = self.mlp_blocks(features)

        if test:  # this is for inference, the batch size is 1
            feat = features / torch.norm(features, dim=-1, keepdim=True)
            feat_similarity = torch.matmul(feat, feat.transpose(1, 2))
            feat_sim = -1 * torch.mean(feat_similarity, dim=2).unsqueeze(2)
            feat_var = torch.var(feat_similarity, dim=2).unsqueeze(2)
            feat_var = torch.mean(feat_var.view(-1, ncrops, t), dim=1)
            feat_sim = torch.mean(feat_sim.view(-1, ncrops, t), dim=1)

        else:
            feat = features / torch.norm(features, dim=-1, keepdim=True)
            normal_feat = feat[0:self.batch_size*ncrops]
            abnormal_feat = feat[self.batch_size*ncrops:]

            afeat_similarity = torch.matmul(abnormal_feat, normal_feat.transpose(1, 2))
            afeat_sim = -1 * torch.mean(afeat_similarity, dim=2).unsqueeze(2)
            afeat_var = torch.var(afeat_similarity, dim=2).unsqueeze(2)
            afeat_var = torch.mean(afeat_var.view(-1, ncrops, t), dim=1)
            afeat_sim = torch.mean(afeat_sim.view(-1, ncrops, t), dim=1)

            nfeat_similarity = torch.matmul(normal_feat, normal_feat.transpose(1, 2))
            nfeat_sim = -1 * torch.mean(nfeat_similarity, dim=2).unsqueeze(2)
            nfeat_var = torch.var(nfeat_similarity, dim=2).unsqueeze(2)
            nfeat_var = torch.mean(nfeat_var.view(-1, ncrops, t), dim=1)
            nfeat_sim = torch.mean(nfeat_sim.view(-1, ncrops, t), dim=1)

            feat_sim = torch.cat((nfeat_sim, afeat_sim), dim=0)
            feat_var = torch.cat((nfeat_var, afeat_var), dim=0)


        scores = features
        scores = self.relu(self.drop_out(self.fc1(scores)))
        scores = self.relu(self.drop_out(self.fc2(scores)))  # [10bs,T,128]
        scores = self.fc3(scores) # [10bs,T,1]
        scores = scores.view(bs, ncrops, -1).mean(1)  # [bs,10,T]->[2bs,T]
        # initialize learnable parameters
        w1 = self.w[0]
        w2 = self.w[1]

        scores = self.sigmoid(w1 * scores + w2 * ( feat_sim + feat_var))

        if test:
            score_abnormal, score_normal, score_select_abnormal, score_select_normal, normal_similarity, abnormal_similarity = 0,0,0,0,0,0
        else:
            normal_scores = scores[0:self.batch_size] #[self.batch_size,32,1]
            abnormal_scores = scores[self.batch_size:] #[self.batch_size,32,1]

            idx_sample_abn = torch.topk(abnormal_scores, self.k_abn, dim=1)[1]
            score_select_abnormal = torch.gather(abnormal_scores, 1, idx_sample_abn)
            score_abnormal = torch.mean(score_select_abnormal, dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude
            #[self.batch_size,32,1]->[self.batch_size,3,1]->[self.batch_size]

            idx_sample_nor = torch.topk(normal_scores, self.k_nor, dim=1)[1]
            score_select_normal = torch.gather(normal_scores, 1, idx_sample_nor)
            score_normal = torch.mean(score_select_normal, dim=1)

            abnormal_similarity = afeat_sim
            normal_similarity = nfeat_sim
            normal_similarity = torch.gather(normal_similarity, 1, idx_sample_nor)

            normal_similarity = torch.mean(normal_similarity,dim=1)
            abnormal_similarity = torch.gather(abnormal_similarity, 1, idx_sample_abn)
            abnormal_similarity = torch.mean(abnormal_similarity, dim=1)

        return score_abnormal, score_normal, scores, score_select_abnormal, score_select_normal, normal_similarity, abnormal_similarity
              #[self.batch_size],           [10*self.batch_size,3,2048],                                          [2self.batch_size,32,1]                    [2self.batch_size,32]       [self.batch_size,3,1]
        




