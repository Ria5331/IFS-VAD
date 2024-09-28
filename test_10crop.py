from re import S
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import sys


def test_mlp(dataloader, model, args, device):
    with torch.no_grad():
        model.eval() #固定参数
        #model.apply(apply_dropout)
        pred = torch.zeros(0) #空tensor

        for i, inputs in enumerate(dataloader):  # i为序列
            input = inputs
            input = input.to(device)

            score_abnormal, score_normal, scores, score_select_abnormal, score_select_normal, normal_similarity, abnormal_similarity = model(input,test=True)

            sig = scores.reshape(-1)
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')  # 标签
        elif args.dataset == 'ucf':
            gt = np.load(f'list/gt-ucf-{args.feature_extractor}.npy')
        else:
            gt = np.load(f'list/gt-xd-{args.feature_extractor}.npy')
        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]

        if args.dataset == 'shanghai' or args.dataset == 'ucf':
            fpr, tpr, threshold = roc_curve(list(gt), pred) #真实标签，预测标签
            np.save('result/fpr.npy', fpr)
            np.save('result/tpr.npy', tpr)
            rec_auc = auc(fpr, tpr)
            print('AUC : ' + str(rec_auc))
            return rec_auc
        else:
            precision, recall, th = precision_recall_curve(list(gt), pred)
            np.save('result/precision.npy', precision)
            np.save('result/recall.npy', recall)
            pr_auc = auc(recall, precision)
            print('AP : ' + str(pr_auc))
            return pr_auc


                   
