from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test_ood_mlp, real_test_ood, test_ood, test_ood_calc, test_viz, test_ood_viz, test_ood_load, test_scores_calc,test_full_ood_mlp,test_mlp
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
from mlp_mixer import MlpModel
from sklearn.metrics import auc, roc_curve
from torchsummary import summary

#viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    #pre_step = 14035 #default = 0
    #pre_auc = 0.9634806689974372 #default = -1

    args = option.parser.parse_args()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    
    #model = Model(args.feature_size, args.batch_size)
    model = MlpModel(1,batch_size=args.batch_size,n_features=args.feature_size)
    #model.load_state_dict( torch.load('./rtfm.pkl'))
    #model.load_state_dict( torch.load('./result/85.52/rtfm147-i3d.pkl'))
    model.load_state_dict( torch.load('./result/95.69/rtfm183-i3d.pkl'))
    
    #if pre_step != 0:
    #    model.load_state_dict( torch.load('./ckpt/' + args.model_name + '{}-i3d.pkl'.format(pre_step)) )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)

    #test_viz(test_loader, model, args, viz ,device)
    #test_ood_viz(test_loader, model, args, viz ,device, 0.32)
    #for i in range(100):
    #    test_ood_load(test_loader, model, args, device,0.01*(i))
    #best lamda = 0.32
    '''
    test_ood_calc(test_loader, model, args, device)
    
    test_scores_calc(test_loader, model, args, device)
    '''
    #test_scores_calc(test_loader, model, args, device)
    #test_ood_calc(test_loader, model, args, device)

    #auc = test_mlp(test_loader, model, args, device)
    #test_ood_mlp(test_loader, model, args, device)

    #summary(model, input_size=(10, 64, 2048))



    #test_ood_load(test_loader, model, args, device, 0.32)
    #test(test_loader, model, args, device)
    #for i in range(100):
    #    test_ood(test_loader, model, args, device,0.01*i)
    auc = test_mlp(test_loader, model, args, device)
    #test_full_ood_mlp(test_loader, model, args, device)
    #real_test_ood(test_loader, model, args, device,0.32)

'''
    pred1 = np.load('./result/score.npy')
    pred2 = np.load('./result/ood.npy')
    gt = np.load('list/gt-ucf.npy')
    fpr, tpr, threshold = roc_curve(list(gt), (pred1 + pred2))
    rec_auc = auc(fpr, tpr)
    print('auc={}'.format(rec_auc))
'''
    #for i in range(100):
        #fpr, tpr, threshold = roc_curve(list(gt), ((1-0.01*i)*pred1 + pred2*0.01*i) )
        #rec_auc = auc(fpr, tpr)
        #print('i={},auc={}'.format(i,rec_auc))



