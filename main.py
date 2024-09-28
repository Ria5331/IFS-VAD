from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from dataset import Dataset
from train import train
from test_10crop import test, test_viz, test_mlp
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
from mlp_mixer import MlpModel
import os
import random
import numpy as np
import time

if __name__ == '__main__':


    #python main.py --feature-extractor clip --dataset ucf --gpus 0

    pre_step = 0
    pre_auc = -1

    args = option.parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpus}'

    config = Config(args)
    config.lr = [0.001]*5000

    if args.dataset == 'shanghai':
        if args.feature_extractor == 'i3d':
            args.feature_size = 2048
        else:
            args.feature_size = 512
        save_auc = 0.97
    elif args.dataset == 'ucf':
        if args.feature_extractor == 'i3d':
            args.feature_size = 2048
            save_auc = 0.84
        else:
            args.feature_size = 512
            save_auc = 0.85
    else: #xd
        save_auc = 0.85
        if args.feature_extractor == 'i3d':
            args.feature_size = 1024
        else:
            args.feature_size = 512

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True,
                               generator = torch.Generator(device='cuda'))
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True,
                               generator = torch.Generator(device='cuda'))
    test_loader = DataLoader(Dataset(args, test_mode=True),
                               batch_size=1, shuffle=False,
                               num_workers=0, pin_memory=False,
                               generator = torch.Generator(device='cuda'))
    model = MlpModel(1, args.batch_size, n_features=args.feature_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=config.lr[0], weight_decay=0.0005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = max(-1,pre_auc)
    output_path = 'result'   # put your own path here

    loadern_iter = iter(train_nloader)
    loadera_iter = iter(train_aloader)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            initial=pre_step,
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        logit = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device, step)

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        if step > 0 and step % 1 == 0 :
            auc = test_mlp(test_loader, model, args, device)
            test_info["epoch"].append(step+pre_step)
            test_info["test_AUC"].append(auc)
            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                if best_AUC >= save_auc:
                    torch.save(model.state_dict(), './result/' + args.model_name + f'-{args.dataset}-' +f'{step}-{args.feature_extractor}.pkl')
                    save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step+pre_step)))
            print('best_auc:'+str(best_AUC))
    

