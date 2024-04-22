from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
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
#viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def seed_everything(seed=2024):
    random.seed(seed)  # ÉèÖÃPythonÄÚÖÃrandomÄ£¿éµÄËæ»úÖÖ×Ó
    os.environ['PYTHONHASHSEED'] = str(seed)  # ÉèÖÃ»·¾³±äÁ¿£¬Ó°ÏìPythonµÄhashº¯Êý
    np.random.seed(seed)  # ÉèÖÃNumPyµÄËæ»úÖÖ×Ó
    torch.manual_seed(seed)  # ÉèÖÃPyTorchµÄËæ»úÖÖ×Ó
    torch.cuda.manual_seed(seed)  # ÉèÖÃPyTorchµÄCUDAËæ»úÖÖ×Ó
    torch.cuda.manual_seed_all(seed)  # ÎªËùÓÐµÄCUDAÉè±¸ÉèÖÃËæ»úÖÖ×Ó
    # Ò»Ð©cudnnµÄ·½·¨¼´Ê¹ÔÚ¹Ì¶¨ÖÖ×ÓºóÒ²¿ÉÄÜÊÇËæ»úµÄ£¬³ý·ÇÄã¸æËßËüÒªÈ·¶¨ÐÔµÄ
    torch.backends.cudnn.benchmark = False  # ¹Ø±ÕcudnnµÄ»ù×¼²âÊÔÄ£Ê½
    torch.backends.cudnn.deterministic = True  # ¿ªÆôcudnnµÄÈ·¶¨ÐÔÄ£Ê½

if __name__ == '__main__':

    seed_everything()
    #torch.multiprocessing.set_start_method("spawn")

    pre_step = 0#210#3581#14035 #default = 0
    pre_auc = -1#0.8569565224125932#-1#0.9551765729882837#0.9634806689974372 #default = -1

    args = option.parser.parse_args()
    config = Config(args)
    config.lr = [0.0001]*5000#0.001

    if args.dataset == 'shanghai': save_auc = 0.95
    elif args.dataset == 'ucf': save_auc = 0.85
    else: save_auc = 0.75

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
    #model = Model(args.feature_size, args.batch_size)
    if args.dataset == 'xd':
        model = MlpModel(1,args.batch_size,n_features=1024)
    else:
        model = MlpModel(1, args.batch_size, n_features=2048)
    
    if pre_step != 0:
        model.load_state_dict( torch.load('./result/85.69/' + args.model_name + '{}-i3d.pkl'.format(pre_step)) )

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.0005)#0.0005

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = max(-1,pre_auc)
    output_path = 'result'   # put your own path here
    #auc = test_mlp(test_loader, model, args, device)
    #test_info["test_AUC"].append(auc)

    loadern_iter = iter(train_nloader)
    loadera_iter = iter(train_aloader)
    #loader_iter = iter(test_loader)

    #auc = test_mlp(loader_iter, model, args, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            initial=pre_step,
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]: #更新学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        
        logit = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device, step)
        
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        #if step % 5 == 0 and step+pre_step > 20:
        auc = test_mlp(test_loader, model, args, device)
        test_info["epoch"].append(step+pre_step)
        test_info["test_AUC"].append(auc)
        
        if test_info["test_AUC"][-1] > best_AUC:
            best_AUC = test_info["test_AUC"][-1]
            if best_AUC >= save_auc:
                torch.save(model.state_dict(), './result/' + args.model_name + '{}-i3d.pkl'.format(step+pre_step))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step+pre_step)))
        print('best_auc:'+str(best_AUC))        
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
    

