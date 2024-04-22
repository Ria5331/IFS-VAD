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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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




class CycleMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 3, dim * 4)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CycleBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

class MlpBlock(nn.Module):
    def __init__(self,input_dim,mlp_dim) :
        super().__init__()
        self.ln=nn.LayerNorm(input_dim)
        self.fc1=nn.Linear(input_dim,mlp_dim)
        self.gelu=nn.GELU()
        self.fc2=nn.Linear(mlp_dim,input_dim)
    
    def forward(self,x):
        #x: (bs,tokens,channels) or (bs,channels,tokens)
        y = self.ln(x)
        y = self.fc1(y)
        y = self.gelu(y)
        y = self.fc2(y)
        y = x+y
        y = self.ln(y)
        
        return y


class LocalCycleMlpBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.global_mlp1 = CycleFC(dim, 2*dim, (4, 1), 1, 0)
        self.global_mlp2 = CycleFC(dim, 2*dim, (32, 1), 1, 0)
        self.global_mlp3 = CycleFC(2*dim, dim, (4, 1), 1, 0)
        self.global_mlp4 = CycleFC(2*dim, dim, (32, 1), 1, 0)
        self.gelu = nn.GELU()
    def forward(self, x):
        # x: (bs,tokens,channels) or (bs,channels,tokens)
        bs, t, f = x.size()
        y = self.ln(x)
        y = (y.permute(0, 2, 1)).reshape(bs, f, t, 1)
        y1 = self.global_mlp1(y)
        y2 = self.global_mlp2(y)
        y = y1 + y2
        #y = torch.cat((y1, y2, y3, y4), dim=1)
        y = self.gelu(y)
        y1 = self.global_mlp3(y)
        y2 = self.global_mlp4(y)
        y = y1 + y2
        #y = torch.cat((y1, y2, y3, y4), dim=1)
        #y = self.gelu(y)
        y = (y.reshape(bs, f, t)).permute(0, 2, 1)
        y = x + y
        y = self.ln(y)

        return y

class GlobalMlpBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.global_mlp1 = nn.Linear(dim, 2*dim)#CycleFC(2048, 2048, (step, 1), 1, 0)
        self.global_mlp2 = nn.Linear(2*dim, dim)#CycleFC(2048, 2048, (step, 1), 1, 0)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        # x: (bs,tokens,channels) or (bs,channels,tokens)
        bs, t, f = x.size()
        y = self.ln(x)
        #y = (y.permute(0, 2, 1)).reshape(bs, f, t, 1)
        y = self.global_mlp1(y)
        y = self.gelu(y)
        y = self.global_mlp2(y)
        #y1 = (y1.reshape(bs, f, t)).permute(0, 2, 1)
        #y2 = (y2.reshape(bs, f, t)).permute(0, 2, 1)
        y = torch.matmul(torch.matmul(x, x.transpose(1, 2)), y)
        #y = self.softmax(y)
        #y = torch.matmul(y, y2)
        y = x + y
        y = self.ln(y)
        return y

'''
class CycleMixer(nn.Module):
    def __init__(self, input, hidden):
        super().__init__()

        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(input)

        self.local_mlp_block1 = CycleFC(input, hidden, (64, 1), 1, 0)
        self.local_mlp_block2 = CycleFC(input, hidden, (16, 1), 1, 0)
        self.local_mlp_block3 = CycleFC(input, hidden, (4, 1), 1, 0)
        self.local_mlp_block4 = CycleFC(hidden, input, (64, 1), 1, 0)
        self.local_mlp_block5 = CycleFC(hidden, input, (16, 1), 1, 0)
        self.local_mlp_block6 = CycleFC(hidden, input, (4, 1), 1, 0)
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(input, hidden)
        self.fc3 = nn.Linear(input, hidden)
        self.fc4 = nn.Linear(hidden, input)
        self.fc5 = nn.Linear(hidden, input)
        self.fc6 = nn.Linear(hidden, input)

    def forward(self, x):
        bs, t, f = x.size()

        local_f = x
        local_f = self.ln(local_f)
        local_f = (local_f.permute(0, 2, 1)).reshape(bs,f,t,1)
        local_f = self.local_mlp_block1(local_f)
        local_f = self.gelu(local_f)
        local_f = self.local_mlp_block4(local_f)
        local_f = (local_f.reshape(bs,f,t)).permute(0,2,1)
        local_f = local_f + x
        x = local_f

        local_f = self.ln(local_f)
        local_f = self.fc1(local_f)
        local_f = self.gelu(local_f)
        local_f = self.fc4(local_f)
        local_f = local_f + x
        x = local_f

        local_f = self.ln(local_f)
        local_f = (local_f.permute(0, 2, 1)).reshape(bs, f, t, 1)
        local_f = self.local_mlp_block2(local_f)
        local_f = self.gelu(local_f)
        local_f = self.local_mlp_block5(local_f)
        local_f = (local_f.reshape(bs, f, t)).permute(0, 2, 1)
        local_f = local_f + x
        x = local_f

        local_f = self.ln(local_f)
        local_f = self.fc2(local_f)
        local_f = self.gelu(local_f)
        local_f = self.fc5(local_f)
        local_f = local_f + x
        x = local_f

        local_f = self.ln(local_f)
        local_f = (local_f.permute(0, 2, 1)).reshape(bs, f, t, 1)
        local_f = self.local_mlp_block3(local_f)
        local_f = self.gelu(local_f)
        local_f = self.local_mlp_block6(local_f)
        local_f = (local_f.reshape(bs, f, t)).permute(0, 2, 1)
        local_f = local_f + x
        x = local_f

        local_f = self.ln(local_f)
        local_f = self.fc3(local_f)
        local_f = self.gelu(local_f)
        local_f = self.fc6(local_f)
        local_f = local_f + x




        return local_f#(bs,n,32,2048)
'''
class CycleMixer(nn.Module):
    def __init__(self, input, hidden):
        super().__init__()

        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(input)
        self.ln2 = nn.LayerNorm(hidden)
        self.local_mlp_block1 = CycleFC(input, hidden, (4, 1), 1, 0)
        self.local_mlp_block2 = CycleFC(input, hidden, (16, 1), 1, 0)
        self.local_mlp_block3 = CycleFC(hidden, input, (4, 1), 1, 0)
        self.local_mlp_block4 = CycleFC(hidden, input, (16, 1), 1, 0)

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
        # local_f1 = (local_f1.reshape(bs,f,t)).permute(0,2,1)
        # local_f2 = (local_f2.reshape(bs,f,t)).permute(0,2,1)
        local_f = local_f * x + x
        local_f = self.ln(local_f)

        return local_f


class MixerBlock(nn.Module):
    def __init__(self,tokens_mlp_dim=16,channels_mlp_dim=1024,tokens_hidden_dim=32,channels_hidden_dim=1024):
        super().__init__()
        self.ln=nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block=MlpBlock(tokens_mlp_dim,mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block=MlpBlock(channels_mlp_dim,mlp_dim=channels_hidden_dim)

    def forward(self,x):
        """
        x: (bs,tokens,channels)
        """
        ### tokens mixing
        y=self.ln(x)
        y=y.transpose(1,2) #(bs,channels,tokens)
        y=self.tokens_mlp_block(y) #(bs,channels,tokens)
        ### channels mixing
        y=y.transpose(1,2) #(bs,tokens,channels)
        out =x+y #(bs,tokens,channels)
        y=self.ln(out) #(bs,tokens,channels)
        y=out+self.channels_mlp_block(y) #(bs,tokens,channels)
        return y

class MlpMixer(nn.Module):
    def __init__(self,num_classes,num_blocks,patch_size,tokens_hidden_dim,channels_hidden_dim,tokens_mlp_dim,channels_mlp_dim):
        super().__init__()
        self.num_classes=num_classes
        self.num_blocks=num_blocks #num of mlp layers
        self.patch_size=patch_size
        self.tokens_mlp_dim=tokens_mlp_dim
        self.channels_mlp_dim=channels_mlp_dim
        self.embd=nn.Conv2d(3,channels_mlp_dim,kernel_size=patch_size,stride=patch_size) 
        self.ln=nn.LayerNorm(channels_mlp_dim)
        self.mlp_blocks=[]
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(tokens_mlp_dim,channels_mlp_dim,tokens_hidden_dim,channels_hidden_dim))
        self.fc=nn.Linear(channels_mlp_dim,num_classes)

    def forward(self,x):
        y=self.embd(x) # bs,channels,h,w
        bs,c,h,w=y.shape
        y=y.view(bs,c,-1).transpose(1,2) # bs,tokens,channels

        if(self.tokens_mlp_dim!=y.shape[1]):
            raise ValueError('Tokens_mlp_dim is not correct.')

        for i in range(self.num_blocks):
            y=self.mlp_blocks[i](y) # bs,tokens,channels
        y=self.ln(y) # bs,tokens,channels
        y=torch.mean(y,dim=1,keepdim=False) # bs,channels
        probs=self.fc(y) # bs,num_classes
        return probs




class MlpModel(nn.Module):
    def __init__(self,num_blocks,batch_size,n_features):
        super().__init__()
        self.num_blocks = num_blocks
        self.mlp_dim = n_features
        self.batch_size = batch_size
        self.num_segments = 64 #ÊÓÆµ·Ö¶Î
        self.k_abn = 5#self.num_segments // 10 #Òì³£top-k==3
        self.k_nor = self.k_abn#self.num_segments // 10 #Õý³£top-k==3
        self.mlp_blocks = CycleMixer(self.mlp_dim, 4096)#TemporalMixe4096rBlock()
        #self.mlp_blocks2 = CycleMixer(self.mlp_dim, 1024)
        #self.mlp_blocks2 = CycleMixer(self.mlp_dim, 4096,16)
        #self.mlp_blocks3 = CycleMixer(self.mlp_dim, 4096,4)
        #self.mlp_blocks4 = CycleMixer()
        #self.mlp_blocks5 = CycleMixer()
        #self.mlp_blocks = CycleMixer_ucf()#TemporalMixerBlock()
        #self.fc = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(self.mlp_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        #self.fc4 = nn.Linear(3, 1)
        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        #self.cycle_fc1 = CycleFC(self.mlp_dim, 512, (2, 1), 1, 0)
        #self.cycle_fc2 = CycleFC(512, 128, (2, 1), 1, 0)
        #self.cycle_fc3 = CycleFC(128, 1, (2, 1), 1, 0)
        #self.ln = nn.LayerNorm(self.mlp_dim)
        #self.conv=nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=4,
        #              stride=1,dilation=1, padding=1)
    
    def forward(self,inputs):
        #y=self.embd(x) # bs,channels,h,w
        #bs,c,h,w=y.shape


        k_abn = self.k_abn
        k_nor = self.k_nor


        bs, ncrops, t, f = inputs.size()
        #print(out.size())
        #print(bs)
        inputs = inputs.reshape(-1, t, f) #[10bs,T,2048]

        #drop_input = torch.ones_like(inputs).cuda()
        #drop_input = self.drop_out2(drop_input)
        #inputs2 = inputs * drop_input

        #print(out.size())
        #if f == 4096:
        #    out = self.fc(out)
        #n = int(t/32)
        #out = out.reshape(-1,n,32,f)
        #out = out.reshape(bs,f,t)
        #out = self.conv(out)
        #out = out.reshape(bs,t,f)

        features = self.mlp_blocks(inputs)
        #features = self.ln(features)
        #features = self.drop_out(features)
        #scores = self.ln(features)
        scores = features
        scores = self.relu(self.drop_out(self.fc1(scores)))
        scores = self.relu(self.drop_out(self.fc2(scores)))#[10bs,T,128]
        logits = self.fc3(scores)#[10bs,T,1]
        logits = logits.view(bs, ncrops, -1).mean(1) #[bs,10,T]->[2bs,T]
        #logits = logits.view(bs, ncrops, -1).transpose(1,2)
        #idx_logits = torch.topk(logits, 1, dim=2)[1]
        #logits = torch.gather(logits, 1, idx_logits)
        #logits = logits.squeeze(2)

        #features = torch.mean(features.view(bs, ncrops, t, f),dim=1)
        normal_features = features[0:self.batch_size*ncrops] #[10self.batch_size,32,2048]
        abnormal_features = features[self.batch_size*ncrops:] #[10self.batch_size,32,2048]
        #n_size = int(normal_features.shape[0]/ncrops)

        feat = features / torch.norm(features, dim=-1, keepdim=True)
        normal_feat = feat[0:self.batch_size*ncrops]
        normal_mean_feat = torch.mean(normal_feat,dim=1,keepdim=True)
        abnormal_feat = feat[self.batch_size*ncrops:]
        #abnormal_mean_feat = torch.mean(abnormal_feat, dim=1, keepdim=True)

        if bs == 1:  # this is for inference, the batch size is 1
            abnormal_feat = normal_feat


        #sim_mask = torch.eye(t).unsqueeze(0)
        #sim_mask = sim_mask + torch.roll(sim_mask,(0,1),dims=(1,2)) + torch.roll(sim_mask,(0,2),dims=(1,2))+ torch.roll(sim_mask, (0, -1), dims=(1, 2)) + torch.roll(sim_mask,(0,-2),dims=(1,2))
        #sim_mask = 1 - sim_mask
        afeat_similarity = torch.matmul(abnormal_feat, normal_feat.transpose(1, 2))
        #afeat_similarity = afeat_similarity * sim_mask
        #abnormal_logits = torch.matmul(afeat_similarity,abnormal_logits.unsqueeze(2))
        #abnormal_logits = self.sigmoid(abnormal_logits)
        #afeat_sim = 1 - afeat_similarity
        #print(afeat_sim.size())
        #numpy.savetxt('afeat_sim.txt', afeat_sim[0].cpu().detach().numpy())
        afeat_sim = 1 - torch.mean(afeat_similarity, dim=2)
        afeat_var = torch.var(afeat_similarity, dim=2)
        afeat_var = torch.mean(afeat_var.view(-1, ncrops, t), dim=1)
        afeat_sim = torch.mean(afeat_sim.view(-1, ncrops, t), dim=1)

        nfeat_similarity = torch.matmul(normal_feat, normal_feat.transpose(1, 2))
        #nfeat_similarity = nfeat_similarity * sim_mask
        #normal_logits = torch.matmul(nfeat_similarity,normal_logits.unsqueeze(2))
        #normal_logits = self.sigmoid(normal_logits)
        nfeat_sim = 1 - torch.mean(nfeat_similarity, dim=2)
        #nfeat_sim = 1 - nfeat_similarity
        nfeat_var = torch.var(nfeat_similarity, dim=2)
        nfeat_var = torch.mean(nfeat_var.view(-1, ncrops, t), dim=1)
        nfeat_sim = torch.mean(nfeat_sim.view(-1, ncrops, t), dim=1)
        #numpy.savetxt('normal_tensor.txt', nfeat_similarity[0][70].cpu().numpy())
        #numpy.savetxt('abnormal_tensor.txt', afeat_similarity[0][30].cpu().numpy())
        #sys.exit()

        if bs == 1:  # this is for inference, the batch size is 1
            feat_sim = nfeat_sim
            feat_var = nfeat_var
        else:
            feat_sim = torch.cat((nfeat_sim, afeat_sim), dim=0)
            feat_var = torch.cat((nfeat_var, afeat_var), dim=0)

        #feat_cosine = feat_sim + feat_var
        #feat_cosine = (feat_cosine - torch.min(feat_cosine)) / (torch.max(feat_cosine) - torch.min(feat_cosine))
        #feat_similarity = torch.matmul(feat, feat.transpose(1, 2))
        #feat_similarity = 1 - torch.mean(feat_similarity, dim=2) #+ torch.var(feat_similarity, dim=2)
        #feat_sim = torch.mean(feat_sim.view(bs, ncrops, -1),dim=1)  # [bs,10,T]->[2bs,T]
        #feat_sim = feat_sim.view(bs, ncrops, -1).transpose(1, 2)
        #idx_feat_sim = torch.topk(feat_sim, 1, dim=2)[1]
        #feat_sim = torch.gather(feat_sim, 1, idx_feat_sim)
        #feat_sim = feat_sim.squeeze(2)

        #feat_var = torch.mean(feat_var.view(bs, ncrops, -1), dim=1)
        #feat_sim = feat_sim.unsqueeze(2)
        #feat_var = feat_var.unsqueeze(2)
        #feat = (feat_sim+feat_var).unsqueeze(2)
        #nfeat = feat[0:self.batch_size]
        #afeat = feat[self.batch_size:]


        feat_magnitudes = torch.norm(features, p=2, dim=2, keepdim=True)/f#¶ÔÎ¬¶È2Çó2·¶Êý[10bs,32,1]
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)#[bs,10,32]->[bs,32]==[2self.batch_size,32]
        #nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes[self.batch_size,32]
        #afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes[self.batch_size,32]

        #feat_sim = (feat_sim - torch.min(feat_sim))/(torch.max(feat_sim)-torch.min(feat_sim))
        #feat_magnitudes = (feat_magnitudes - torch.min(feat_magnitudes))/(torch.max(feat_magnitudes)-torch.min(feat_magnitudes))
        #feat = features / torch.norm(features, dim=-1, keepdim=True)
        #mean_feature = features.mean(dim=1,keepdim=True)
        #mean_feat = mean_feature / torch.norm(mean_feature, dim=-1, keepdim=True)
        #feat_similarity = torch.matmul(feat, mean_feat.transpose(1,2))
        #feat_similarity = torch.mean(feat_similarity,dim=2,keepdim=True)

        #scores = torch.cat((logits.unsqueeze(2),feat_sim.unsqueeze(2),feat_var.unsqueeze(2)),dim=2)
        #scores = self.fc4(scores)

        #scores = self.sigmoid(logits).unsqueeze(2)
        #scores = logits.unsqueeze(2)

        scores = (self.sigmoid(logits) + self.sigmoid(feat_sim + feat_var))/2
        scores = scores.unsqueeze(2)
        logits = (self.sigmoid(logits) + self.sigmoid(feat_sim + feat_var))/2
        logits = logits.unsqueeze(2)
        #print(feat_magnitudes)

        #numpy.savetxt('scores.txt', afeat_similarity[0][80].cpu().numpy())
        #sys.exit()
        #scores = ((self.sigmoid(logits) + + self.sigmoid(1 - feat_sim + reconstruct/64)) / 2).unsqueeze(2)

        #scores = ((self.sigmoid(logits) + (1-feat_sim)/2 + feat_var)/3).unsqueeze(2)
        #scores2 = torch.roll(scores,1,1)
        #scores3 = torch.roll(scores,-1,1)
        #scores4 = torch.roll(scores, 2, 1)
        #scores5 = torch.roll(scores, -2, 1)
        #scores = (scores + scores2 + scores3 + scores4 + scores5)/5
        afeat_sim = afeat_sim.unsqueeze(2)
        nfeat_sim = nfeat_sim.unsqueeze(2)

        nfea_magnitudes = feat_magnitudes[0:self.batch_size].unsqueeze(2)  # normal feature magnitudes[self.batch_size,32]
        afea_magnitudes = feat_magnitudes[self.batch_size:].unsqueeze(2)  # abnormal feature magnitudes[self.batch_size,32]

        normal_scores = scores[0:self.batch_size] #[self.batch_size,32,1]
        abnormal_scores = scores[self.batch_size:] #[self.batch_size,32,1]

        normal_logits = logits[0:self.batch_size]
        abnormal_logits = logits[self.batch_size:]


        #logits = (reconstruct/2048).unsqueeze(2)
        #logits = ((self.sigmoid(logits) + self.sigmoid(1-feat_sim))/2).unsqueeze(2)

        #logits = self.sigmoid(logits).unsqueeze(2)
        #logits = feat_similarity[0:self.batch_size]
        #logits = scores
        #logits = logits.unsqueeze(2)


        if bs == 1:  # this is for inference, the batch size is 1
            abnormal_features = normal_features
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            #afeat = nfeat
            abnormal_logits = normal_logits

        alogits = abnormal_scores # afea_magnitudes.unsqueeze(2)#1-afea_similarity.unsqueeze(2)+0.01*afea_magnitudes.unsqueeze(2)#+abnormal_scores#+afea_magnitudes.unsqueeze(2)#+abnormal_scores#+afea_similarity.unsqueeze(2)#+afea_magnitudes.unsqueeze(2)#abnormal_variance.view(math.ceil(bs/2), ncrops, t, 1).mean(1)*abnormal_mean.view(math.ceil(bs/2), ncrops, t, 1).mean(1)#*abnormal_logits.unsqueeze(2)#*abnormal_variance_feat.view(math.ceil(bs/2), ncrops, t, 1).mean(1)#*afea_magnitudes.unsqueeze(2)#abnormal_variance_feat.view(math.ceil(bs/2), ncrops, t, 1).mean(1)#+1-afea_similarity.unsqueeze(2)#afea_magnitudes.unsqueeze(2) #torch.norm(afea_similarity,p=2,dim=2,keepdim=True)
        nlogits = normal_scores # nfea_magnitudes.unsqueeze(2)#1-nfea_similarity.unsqueeze(2)+0.01*nfea_magnitudes.unsqueeze(2)#+normal_scores#+afea_magnitudes.unsqueeze(2)#+normal_scores#+nfea_similarity.unsqueeze(2)#+nfea_magnitudes.unsqueeze(2)#normal_variance.view(math.ceil(bs/2), ncrops, t, 1).mean(1)*normal_mean.view(math.ceil(bs/2), ncrops, t, 1).mean(1)#*normal_logits.unsqueeze(2)#*normal_variance_feat.view(math.ceil(bs/2), ncrops, t, 1).mean(1)#*nfea_magnitudes.unsqueeze(2)#normal_variance_feat.view(math.ceil(bs/2), ncrops, t, 1).mean(1)#+1-nfea_similarity.unsqueeze(2)#nfea_magnitudes.unsqueeze(2)#torch.norm(nfea_similarity,p=2,dim=2,keepdim=True)

        '''
        alogit_flatten = alogits.reshape(-1)
        abnormal_scores_flatten = abnormal_scores.reshape(-1)
        mask_select_abnormal_sample = torch.zeros_like(alogits, dtype=torch.bool)
        topk_abnormal_sample = torch.topk(alogits, k_abn, dim=1)[1]
        mask_select_abnormal_sample.scatter_(1, topk_abnormal_sample, True)

        mask_select_abnormal_batch = torch.zeros_like(alogit_flatten, dtype=torch.bool)
        topk_abnormal_batch = torch.topk(alogit_flatten, int(k_abn*bs/2))[1]
        mask_select_abnormal_batch.scatter_(0, topk_abnormal_batch, True)

        mask_select_abnormal = mask_select_abnormal_sample.reshape(-1) | mask_select_abnormal_batch
        score_select_abnormal = torch.masked_select(abnormal_scores_flatten, mask_select_abnormal)
        score_abnormal = torch.mean(score_select_abnormal)
        total_select_abn_feature = abnormal_features
        alogit_select = alogits
        '''


        #abnormal_logits_flatten = abnormal_logits.reshape(-1)
        #abnormal_scores_flatten = abnormal_scores.reshape(-1)
        #abnormal_logits_drop = ((afea_magnitudes+1-alogit).unsqueeze(2)) * select_idx
        idx_sample_abn = torch.topk(abnormal_scores, k_abn, dim=1)[1]
        #idx_batch_abn = torch.topk(abnormal_scores_flatten, int(k_abn*bs/4))[1]
        score_select_abnormal = torch.gather(abnormal_logits, 1, idx_sample_abn)
        #score_select_batch_abnormal = torch.gather(abnormal_logits_flatten, 0, idx_batch_abn)
        #score_batch_abnormal = score_select_batch_abnormal
        score_abnormal = torch.mean(score_select_abnormal, dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude
        #[self.batch_size,32,1]->[self.batch_size,3,1]->[self.batch_size]
        #score_abnormal = (score_abnormal + score_batch_abnormal)/2

        afeat_sim_select = torch.gather(afeat_sim, 1, idx_sample_abn)

        feat_select_abnormal = torch.gather(afea_magnitudes, 1, idx_sample_abn)



        ####### process normal videos -> select top3 feature magnitude #######
        #Óë´¦ÀíÒì³£ÊÓÆµ²½Öè»ù±¾ÏàÍ¬,»ñµÃµ¥¸öÊÓÆµµÄÒì³£·ÖÊý¼°3¸ö×î´óµÄÕý³£ÌØÕ÷·ùÖµ

        #select_idx_normal = torch.ones_like(nlogit).cuda()
        #select_idx_normal = self.drop_out(select_idx_normal)

        #nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        #idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        #idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])
        '''

        nlogit_flatten = nlogit.reshape(-1)
        normal_scores_flatten = normal_scores.reshape(-1)
        mask_select_normal_batch = torch.zeros_like(nlogit_flatten, dtype=torch.bool)
        topk_normal_batch = torch.topk(nlogit_flatten, num_select_abnormal)[1]
        mask_select_normal_batch.scatter_(0, topk_normal_batch, True)

        mask_select_normal = mask_select_normal_batch
        score_select_normal = normal_scores_flatten[mask_select_normal]
        score_normal = score_select_normal

        total_select_nor_feature = abnormal_features
        nlogit_select = nlogit
        

        '''

        normal_logits_flatten = normal_logits.reshape(-1)
        normal_scores_flatten = normal_scores.reshape(-1)
        #abnormal_logits_drop = ((afea_magnitudes+1-alogit).unsqueeze(2)) * select_idx
        idx_sample_nor = torch.topk(normal_scores, k_nor, dim=1)[1]
        idx_batch_nor = torch.topk(normal_scores_flatten, int(k_nor*bs/4))[1]
        score_select_normal = torch.gather(normal_logits, 1, idx_sample_nor)
        score_select_batch_normal = torch.gather(normal_logits_flatten, 0, idx_batch_nor)
        score_batch_normal = score_select_batch_normal
        score_normal = torch.mean(score_select_normal, dim=1)

        nfeat_sim_select = torch.gather(nfeat_sim, 1, idx_sample_nor)

        feat_select_normal = torch.gather(nfea_magnitudes, 1, idx_sample_nor)
        '''
        normal_logits_drop = (nlogits)# * select_idx_normal
        #normal_logits_drop = ((nfea_magnitudes+1-nlogit).unsqueeze(2)) * select_idx
        idx_normal = torch.topk(normal_logits_drop, k_nor, dim=1)[1] #È¡top3Òì³£·ùÖµ,topk[0]·µ»ØÖµ,topk[1]·µ»ØË÷Òý,[self.batch_size,3]
        #idx_abn = torch.topk(abnormal_logits_drop, k_abn * self.batch_size, dim=0)[1]
        #nlogit_drop = nlogit * select_idx_normal
        #idx_normal = torch.topk(nlogit_drop, k_nor, dim=1)[1]

        #score_select_normal = torch.gather(normal_scores.view(-1), 0, idx_normal)
        #score_normal = torch.mean(score_select_normal, dim=0)
        #score_normal = score_normal.view(-1,1)

        #idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])
        #idx_normal_feat = idx_normal.expand([-1, -1, normal_features.shape[2]])
        #normal_features = normal_features.view(n_size, ncrops, t, f)
        #normal_features = normal_features.permute(1, 0, 2, 3)

        #total_select_nor_feature = torch.zeros(0)
        #for nor_fea in normal_features:
        #    feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
        #    total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        #idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        #idx_normal_score = idx_normal.expand([-1, -1, normal_logits.shape[2]])
        score_select_normal = torch.gather(normal_logits, 1, idx_normal)
        score_normal = torch.mean(score_select_normal, dim=1) # top 3 scores in normal bag
        #score_normal = torch.mean(score_normal)

        #nlogit_select = torch.gather(nlogit,1,idx_normal.squeeze(2))
        nlogit_select = torch.gather(nlogits, 1, idx_normal)
        '''
        #alogit_select = alogit
        #nlogit_select = nlogit
        #feat_select_abn = alogits#total_select_abn_feature#alogit#
        #feat_select_normal = nlogits#total_select_nor_feature#nlogit#

        return score_abnormal, score_normal, feat_select_abnormal, feat_select_normal, afeat_sim_select, nfeat_sim_select, scores, abnormal_logits, normal_logits, feat_magnitudes,score_select_abnormal,score_select_normal,alogits,logits
              #[self.batch_size],           [10*self.batch_size,3,2048],                                          [2self.batch_size,32,1]                    [2self.batch_size,32]       [self.batch_size,3,1]
        

if __name__ == '__main__':
    mlp_mixer=MlpMixer(num_classes=1000,num_blocks=10,patch_size=10,tokens_hidden_dim=32,channels_hidden_dim=1024,tokens_mlp_dim=16,channels_mlp_dim=1024)
    input=torch.randn(50,3,40,40)
    output=mlp_mixer(input)
    print(output.shape)




