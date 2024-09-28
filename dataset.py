import torch.utils.data as data
import numpy as np
from utils import process_feat,process_split
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, list_file=None):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.feature_extractor = args.feature_extractor
        if list_file == None :
            if self.dataset == 'shanghai':
                if test_mode:#test
                    if self.feature_extractor == 'clip':
                        self.rgb_list_file = f'list/shanghai-{self.feature_extractor}-test.list'
                    else:
                        self.rgb_list_file = f'list/shanghai-{self.feature_extractor}-test-10crop.list'
                else:#train
                    if self.feature_extractor == 'clip':
                        self.rgb_list_file = f'list/shanghai-{self.feature_extractor}-train.list'
                    else:
                        self.rgb_list_file = f'list/shanghai-{self.feature_extractor}-train-10crop.list'
            elif self.dataset == 'ucf':
                if test_mode:#test
                    self.rgb_list_file = f'list/ucf-{self.feature_extractor}-test-10crop.list'
                else:#train
                    self.rgb_list_file = f'list/ucf-{self.feature_extractor}-train-10crop.list'
            elif self.dataset == 'xd':
                if test_mode:#test
                    if args.feature_extractor == 'i3d':
                        self.rgb_list_file = f'list/xd-{self.feature_extractor}-test-5crop.list'
                    else:
                        self.rgb_list_file = f'list/xd-{self.feature_extractor}-test-10crop.list'
                else:#train
                    if args.feature_extractor == 'i3d':
                        self.rgb_list_file = f'list/xd-{self.feature_extractor}-train-5crop.list'
                    else:
                        self.rgb_list_file = f'list/xd-{self.feature_extractor}-train-10crop.list'
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
                else:#abnormal
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
            
            elif self.dataset == 'xd':
                if self.is_normal:
                    self.list = self.list[1905:]
                    print('normal list for xd')
                else:
                    self.list = self.list[:1905]
                    print('abnormal list for xd')


    def __getitem__(self, index):
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        label = self.get_label()  # get video level label 0/1
        features = np.array(features, dtype=np.float32)
        if self.dataset == 'shanghai' and self.feature_extractor == 'clip':
            features = features.reshape(features.shape[0],1,features.shape[1])
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            features = features.transpose(1, 0, 2)  # [T,N,F]->[N,T,F]
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2) #[T,N,F]->[N,T,F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 64)  # divide a video into 32 segments
                #feature = process_split(feature, 64)
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
