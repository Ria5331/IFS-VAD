import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, list_file=None):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if list_file == None :
            if self.dataset == 'shanghai':
                if test_mode:#test
                    self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
                else:#train
                    self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
            elif self.dataset == 'ucf':
                if test_mode:#test
                    self.rgb_list_file = 'list/ucf-i3d-test-10crop.list'
                else:#train
                    self.rgb_list_file = 'list/ucf-i3d-train-10crop.list'
            elif self.dataset == 'xd':
                if test_mode:#test
                    self.rgb_list_file = 'list/xd-i3d-test.list'
                else:#train
                    self.rgb_list_file = 'list/xd-i3d-train.list'
        else:
            self.rgb_list_file = list_file

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:#train
            if self.dataset == 'shanghai':
                if self.is_normal:#normal
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:#abnormal
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)
            
            elif self.dataset == 'xd':
                if self.is_normal:
                    self.list = self.list[9525:]
                    print('normal list for xd')
                    #print(self.list)
                else:
                    self.list = self.list[:9525]
                    print('abnormal list for xd')
                    #print(self.list)


    def __getitem__(self, index):
        if self.dataset == 'xd':
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = features.reshape(-1, 1, 1024)
        else:
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)

        label = self.get_label()  # get video level label 0/1
        features = np.array(features, dtype=np.float32)  # 获取I3D预处理特征并转换为float32格式

        if self.tranform is not None:#如果有预设的transform
            features = self.tranform(features)
        if self.test_mode:#如果是test就不进行数据增强
            features = features.transpose(1, 0, 2) #[T,N,F]->[N,T,F]
            return features #[N-CROP,T,F]
        else:
            # process 10-cropped snippet feature

            features = features.transpose(1, 0, 2) #[T,N,F]->[N,T,F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 64)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32) #[N,32,F]     
            return divided_features, label #[N,32,F], 1

    def get_label(self):

        if self.is_normal:#normal=0
            label = torch.tensor(0.0)
        else:#abnormal=1
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
