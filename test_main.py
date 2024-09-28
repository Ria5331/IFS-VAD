from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from dataset import Dataset
from train import train
from test_10crop import test_ood_mlp, real_test_ood, test_ood, test_ood_calc, test_viz, test_ood_viz, test_ood_load, test_scores_calc,test_full_ood_mlp,test_mlp
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
from mlp_mixer import MlpModel
from sklearn.metrics import auc, roc_curve

if __name__ == '__main__':

    #python test_main.py --feature-extractor clip --dataset ucf --gpus 0

    args = option.parser.parse_args()

    if args.dataset == 'shanghai':
        if args.feature_extractor == 'i3d':
            args.feature_size = 2048
        else:
            args.feature_size = 512
    elif args.dataset == 'ucf':
        if args.feature_extractor == 'i3d':
            args.feature_size = 2048
        else:
            args.feature_size = 512
    else: #xd
        if args.feature_extractor == 'i3d':
            args.feature_size = 1024
        else:
            args.feature_size = 512

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)


    model = MlpModel(1, batch_size=args.batch_size, n_features=args.feature_size)

    #model.load_state_dict(torch.load('./result/97.95/IFS-shanghai-clip.pkl'))
    model.load_state_dict(torch.load('./result/86.57/IFS-ucf-clip.pkl'))
    #model.load_state_dict( torch.load('./result/83.14/IFS-xd-clip.pkl'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    auc = test_mlp(test_loader, model, args, device)



