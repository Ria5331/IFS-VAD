from re import S
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def ViM(feature,logit,device,ncrops = 10):#feature==[10self.batch_size,T,2048],logit==[10self.batch_size,T,1]
    #print(logit.size())
    logit = logit.squeeze(2)#[10self.batch_size,T]
    batch_size = int(feature.size()[0]/ncrops)
    T = feature.size()[1]
    #int(feature.size()[0]/batch_size)
    #主空间保留的子项D
    DIM = 1000 #if feature.shape[-1] >= 2048 else 512
    #计算主空间
    #print(logit.size())
    logit = logit.view(1,batch_size*T,-1)#[1,self.batch_size*T,1]
    logit = logit.permute(1,0,2) #[self.batch_size*T,10,1]
    logit = torch.mean(logit,dim=1).squeeze()#[batch_size*T]
    feature = feature.view(ncrops,batch_size*T,-1)
    feature = feature.permute(1,0,2) 
    feature = torch.mean(feature,dim=1).squeeze()#[batch_size*T,2048]
    feat_mean = torch.mean(feature,dim = 0)#[1,2048]
    #ec = EmpiricalCovariance(assume_centered=True)
    #ec.fit(feature)
    feature_cov = torch.matmul((feature - feat_mean).T,feature - feat_mean)
    #feature_cov = pearsonr(feature - feat_mean,feature - feat_mean)
    #print(feature_cov.size())
    #eig_vals, eigen_vectors = torch.eig(feature_cov,eigenvectors=True)
    eig_vals, eigen_vectors = torch.linalg.eig(feature_cov)
    eig_vals = eig_vals.type(torch.FloatTensor)
    eigen_vectors = eigen_vectors.type(torch.FloatTensor)
    #U, S, V = torch.svd(feature - feat_mean)
    #np.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    #np.argsort对数组从小到大排序，返回排序后的原数组序列号,乘以-1后从大到小排序
    # N*N(F) -> (F)N*N -> (F)(N-D)*N -> N*(N-D)(F) == R
    #NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    #torch.mm(feature - feat_mean, U[:,:,DIM])
    NS = (eigen_vectors.T[torch.sort(eig_vals,descending=True)[DIM:]]).T
    NS = NS.to(device)
    #feature.to(device)
    #feat_mean.to(device)

    #计算alpha
    energy = torch.logsumexp(logit, dim=-1)

    vlogit = torch.norm(torch.matmul(feature - feat_mean, NS),dim=-1)
    alpha = torch.mean(torch.max(logit),dim=-1) / torch.mean(vlogit)
    #print(vlogit.size())
    #print(energy.size())
    #print(logit.size())
    #print(f'{alpha:.4f}')
    ood_score = (alpha*vlogit - energy).view(T,batch_size)
    ood_score = torch.mean(ood_score,dim=1).squeeze().T
    return ood_score#[T]

def PSA(id_feature,id_logit):
    id_logit = id_logit.squeeze(2)#[self.batch_size,T]
    batch_size,T = id_logit.size()
    ncrops = int(id_feature.size()[0]/batch_size)
    #主空间保留的子项D
    DIM = 1000 #if id_feature.shape[-1] >= 2048 else 512
    #计算主空间
    id_feature = id_feature.view(ncrops,batch_size*T,-1)
    id_feature = id_feature.permute(1,0,2) 
    id_feature = torch.mean(id_feature,dim=1).squeeze()#[batch_size*T,2048]
    feat_mean = torch.mean(id_feature,dim = 0)#[1,2048]
    #ec = EmpiricalCovariance(assume_centered=True)
    #ec.fit(id_feature)
    id_feature_cov = torch.matmul((id_feature - feat_mean).T,id_feature - feat_mean)
    #id_feature_cov = pearsonr(id_feature - feat_mean,id_feature - feat_mean)
    #print(id_feature_cov.size())
    eig_vals, eigen_vectors = torch.eig(id_feature_cov,eigenvectors=True)
    #U, S, V = torch.svd(id_feature - feat_mean)
    #np.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    #np.argsort对数组从小到大排序，返回排序后的原数组序列号,乘以-1后从大到小排序
    # N*N(F) -> (F)N*N -> (F)(N-D)*N -> N*(N-D)(F) == R
    #NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    #torch.mm(id_feature - feat_mean, U[:,:,DIM])
    NS = (eigen_vectors.T[torch.sort(eig_vals,descending=True)[DIM:]]).T
    return NS

def OOD(feature,logit,NS):
    logit = logit.squeeze(2)#[self.batch_size,T]
    batch_size,T = logit.size()
    ncrops = int(feature.size()[0]/batch_size)
    feature = feature.view(ncrops,batch_size*T,-1)
    feature = feature.permute(1,0,2) 
    feature = torch.mean(feature,dim=1).squeeze()#[batch_size*T,2048]
    feat_mean = torch.mean(feature,dim = 0)#[1,2048]
    energy = torch.logsumexp(logit, dim=-1)
    vlogit = torch.norm(torch.matmul(feature - feat_mean, NS),dim=-1)
    alpha = torch.mean(torch.max(logit),dim=-1) / torch.mean(vlogit)
    #print(f'{alpha:.4f}')
    ood_score = (alpha*vlogit - energy).view(T,batch_size)
    ood_score = torch.mean(ood_score,dim=1).squeeze().T
    return ood_score

def test_viz(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval() #固定参数
        pred = torch.zeros(0) #空tensor

        for i, input in enumerate(dataloader): #i为序列
            input = input.to(device)
            input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes,score_select_abn,score_select_nor,feats,logits = model(input) #logits为scores
            print(scores)
            #vim_score = ViM(feats,logits)
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            sig = scores#*0.8 + vim_score*0.2
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #标签
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        fpr, tpr, threshold = roc_curve(list(gt), pred) #真实标签，预测标签
        np.save('result/fpr.npy', fpr)
        np.save('result/tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('result/precision.npy', precision)
        np.save('result/recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc

def test_viz_mlp(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval() #固定参数
        pred = torch.zeros(0) #空tensor

        for i, inputs in enumerate(dataloader): #i为序列
            input,t = inputs
            input = input.to(device)
            
            #input = input.permute(0, 2, 1, 3)
            #input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, score_select_abnormal, score_select_normal,norm_feat,logits = model(input)  
            #vim_score = ViM(feats,logits)
            
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            scores = scores[:t]
            sig = scores
            pred = torch.cat((pred, sig))
            

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #标签
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        
        fpr, tpr, threshold = roc_curve(list(gt), pred) #真实标签，预测标签
        np.save('result/fpr.npy', fpr)
        np.save('result/tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('result/precision.npy', precision)
        np.save('result/recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc
    
    
def gaussian_filter(size, sigma):
    if type(size) == int:
        siz = int((size - 1) / 2)
        x = np.array(range(-siz, siz + 1))
        arg = -x ** 2 / (2 * sigma ** 2)
        h = np.exp(arg)
        sumh = np.sum(h)
        h = h / sumh
    elif type(size) == list:
        siz0 = int((size[0] - 1) / 2)
        siz1 = int((size[1] - 1) / 2)
        x = np.array(range(-siz0, siz0 + 1))
        y = np.array(range(-siz1, siz1 + 1))
        [x, y] = np.meshgrid(x, y)
        arg = -(x ** 2 + y ** 2) / (2 * sigma ** 2)
        h = np.exp(arg)
        sumh = np.sum(h)
        h = h / sumh
    return np.round(h, 4)

def test_mlp(dataloader, model, args, device):
    with torch.no_grad():
        model.eval() #固定参数
        #model.apply(apply_dropout)
        pred = torch.zeros(0) #空tensor

        for i, inputs in enumerate(dataloader):  # i为序列
            inputs = inputs.to(device)  # [b,T,10,F] -> [b,10,T,F]

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, alogit_select, \
                nlogit_select, scores, alogit, nlogit, feat_magnitudes, score_select_abnormal, score_select_normal, score_batch_abnormal, logits = model(
                inputs)

            scores = logits
            scores = torch.squeeze(scores, 2)  # [b,T,1]->[b,T]
            scores = torch.mean(scores, 0)  # [T]
            sig = scores
            #print(scores.size())
            pred = torch.cat((pred, sig))

        if args.dataset == 'xd':
            pred = pred.reshape(-1, 5)
            pred = torch.mean(pred, dim=1)

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')  # 标签
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        else:
            gt = np.load('list/gt-xd.npy')
        #np.savetxt('gt.txt', gt)
        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        #pred = pred/np.max(pred)
        #pred_drop = np.load('scores_drop.npy')
        #difference = np.maximum(pred - pred_drop,0)
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        #f = gaussian_filter(32, 5)
        #pred = np.convolve(pred, f, mode="same")
        #difference = pred_drop - pred
        #pred = pred-difference
        #np.save('scores.npy', pred)
        #np.save('scores_drop.npy', pred)
        #np.savetxt('scores.txt', pred[142448:])
        #np.savetxt('gt.txt', gt[142448:])
        #sys.exit()
        #print(gt.shape)
        #print(pred.shape)
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






def test_ood_mlp(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()  # 固定参数
        # model.apply(apply_dropout)
        pred1 = torch.zeros(0)  # 空tensor
        pred2 = torch.zeros(0)
        if args.dataset == 'xd':
            features = torch.tensor([])
            for i, inputs in enumerate(dataloader):  # i为序列
                # inputs = inputs.to(device)
                features = torch.cat((features, inputs), dim=0)
                if (i % 5) == 4:
                    features = features.to(device)

                    # input = input.permute(0, 2, 1, 3)
                    # input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
                    score_abnormal, score_normal, feat_select_abn, feat_select_normal, alogit_select, \
                        nlogit_select, scores, alogit, nlogit, feat_magnitudes, score_select_abnormal, score_select_normal, feats, logits = model(
                        inputs)

                    #scores = logits  # scores
                    scores = torch.squeeze(scores, 2)  # [b,T,1]->[b,T]
                    scores = torch.mean(scores, 0)  # [T]
                    # scores = scores[:t]
                    vim_score = ViM(feats, logits, device)
                    sig1 = scores
                    pred1 = torch.cat((pred1, sig1))
                    sig2 = vim_score
                    pred2 = torch.cat((pred2, sig2))
                    features = torch.tensor([])
        else:
            for i, inputs in enumerate(dataloader):  # i为序列
                inputs = inputs.to(device)
                # input = input.permute(0, 2, 1, 3)
                # input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
                score_abnormal, score_normal, feat_select_abn, feat_select_normal, alogit_select, \
                    nlogit_select, scores, alogit, nlogit, feat_magnitudes, score_select_abnormal, score_select_normal, feats, logits = model(
                    inputs)

                scores = logits #scores
                scores = torch.squeeze(scores, 2)  # [b,T,1]->[b,T]
                scores = torch.mean(scores, 0)  # [T]
                # scores = scores[:t]
                vim_score = ViM(feats, logits, device)
                sig1 = scores
                pred1 = torch.cat((pred1, sig1))
                sig2 = vim_score
                pred2 = torch.cat((pred2, sig2))


        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #标签
        else:
            gt = np.load('list/gt-ucf.npy')

        pred1 = list(pred1.cpu().detach().numpy())  # [T*batch_num]
        pred1 = np.repeat(np.array(pred1), 16)  # [T*batch_num*16]

        pred2 = list(pred2.cpu().detach().numpy())  # [T*batch_num]
        pred2 = np.repeat(np.array(pred2), 16)  # [T*batch_num*16]

        np.save('./scores.npy', pred1)
        np.save('./vim.npy', pred2)
        for i in range(100):
            fpr, tpr, threshold = roc_curve(list(gt), pred1 * (1 - 0.01 * i) + pred2 * 0.01 * i)
            rec_auc = auc(fpr, tpr)
            print('{}*scores + {}*vim : auc={}'.format(1 - 0.01 * i, 0.01 * i, rec_auc))

        #precision, recall, th = precision_recall_curve(list(gt), pred)
        #pr_auc = auc(recall, precision)
        #np.save('result/precision.npy', precision)
        #np.save('result/recall.npy', recall)
        #return rec_auc

def test_full_ood_mlp(dataloader, model, args, device):
    with torch.no_grad():
        model.eval() #固定参数
        pred1 = torch.zeros(0) #空tensor
        pred2 = torch.zeros(0) #空tensor
        for i, inputs in enumerate(dataloader): #i为序列
            input,t = inputs
            input = input.to(device)
            
            #input = input.permute(0, 2, 1, 3)
            #input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, score_select_abnormal, score_select_normal,feats,logits = model(input)  
            vim_score = ViM(feats,logits)
            
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            scores = scores[:t]
            vim_score = vim_score[:t]
            #sig = scores*(1-lamda) + vim_score*lamda
            sig1 = scores
            pred1 = torch.cat((pred1, sig1))
            sig2 = vim_score
            pred2 = torch.cat((pred2, sig2))
            

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #标签
        else:
            gt = np.load('list/gt-ucf.npy')

        pred1 = list(pred1.cpu().detach().numpy()) #[T*batch_num]
        pred1 = np.repeat(np.array(pred1), 16) #[T*batch_num*16]

        pred2 = list(pred2.cpu().detach().numpy()) #[T*batch_num]
        pred2 = np.repeat(np.array(pred2), 16) #[T*batch_num*16]
        
        #fpr, tpr, threshold = roc_curve(list(gt), pred) #真实标签，预测标签
        np.save('./scores.npy', pred1)
        np.save('./vim.npy', pred2)
        for i in range(100):
            fpr, tpr, threshold = roc_curve(list(gt), pred1*(1-0.01*i) + pred2*0.01*i)
            rec_auc = auc(fpr, tpr)
            print('i={},auc={}'.format(i,rec_auc))


        #rec_auc = auc(fpr, tpr)
        #print('auc : ' + str(rec_auc))

        #precision, recall, th = precision_recall_curve(list(gt), pred)
        #pr_auc = auc(recall, precision)
        #np.save('result/precision.npy', precision)
        #np.save('result/recall.npy', recall)
        #return rec_auc

def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval() #固定参数
        pred = torch.zeros(0) #空tensor

        for i, input in enumerate(dataloader): #i为序列
            input = input.to(device)
            input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, score_select_abnormal, score_select_normal,norm_feat,logits = model(input)  
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            sig = scores
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #标签
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        fpr, tpr, threshold = roc_curve(list(gt), pred) #真实标签，预测标签
        rec_auc = auc(fpr, tpr)
        print('auc = ' + str(rec_auc))
        return rec_auc


def test_ood(dataloader, model, args, device,lamda):
    with torch.no_grad():
        model.eval() #固定参数
        pred = torch.zeros(0) #空tensor

        for i, inputs in enumerate(dataloader): #i为序列
            input,t = inputs
            input = input.to(device)
            input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes,score_select_abn,score_select_nor,feats,logits = model(inputs=input) #logits为scores
            vim_score = ViM(feats,scores)
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            scores = scores[:t]
            
            sig = scores*(1-lamda) + vim_score*lamda
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #标签
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        fpr, tpr, threshold = roc_curve(list(gt), pred) #真实标签，预测标签
        rec_auc = auc(fpr, tpr)
        print('lamda = '+ str(lamda) +', ood_auc = ' + str(rec_auc))
        return rec_auc

def test_ood_viz(dataloader, model, args, viz, device, lamda):
    with torch.no_grad():
        model.eval() #固定参数
        pred = torch.zeros(0) #空tensor

        for i, input in enumerate(dataloader): #i为序列
            input = input.to(device)
            input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes,score_select_abn,score_select_nor,feats,logits = model(inputs=input) #logits为scores
            vim_score = ViM(feats,scores)
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            sig = scores*(1-lamda) + vim_score*lamda
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #标签
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        fpr, tpr, threshold = roc_curve(list(gt), pred) #真实标签，预测标签
        rec_auc = auc(fpr, tpr)
        viz.lines('roc', tpr, fpr)
        print('lamda = '+ str(lamda) +', ood_auc = ' + str(rec_auc))
        return rec_auc

def test_ood_load(dataloader, model, args, device, lamda):
    if args.dataset == 'shanghai':
        gt = np.load('list/gt-sh.npy') #标签
    else:
        gt = np.load('list/gt-ucf.npy')

    vim  = np.load('result/ood.npy')
    pred = np.load('result/score.npy')
    pred = 1.5+(pred * (1-lamda) + vim*lamda)
    pred_max = max(pred)
    print(pred_max)
    pred_min = min(pred)
    print(pred_min)
    fpr, tpr, thresholds = roc_curve(list(gt), pred) #真实标签，预测标签
    roc_auc = auc(fpr, tpr)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    #print('lamda = '+ str(lamda) +', ood_auc = ' + str(rec_auc))
    #print(threshold)
    print(100*roc_auc)
    #print(threshold)
    #print(thresholds)


def test_ood_calc(dataloader, model, args, device):
    with torch.no_grad():
        model.eval() #¹Ì¶¨²ÎÊý
        pred = torch.zeros(0) #¿Õtensor

        for i, inputs in enumerate(dataloader): #iÎªÐòÁÐ
            input,t = inputs
            input = input.to(device)
            #input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, score_select_abnormal, score_select_normal,feats,logits = model(input)  
            #print(feats.size())
            #print(scores.size())
            vim_score = ViM(feats,scores)
            #scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            #scores = torch.mean(scores, 0) #[T]
            #sig = scores*(1-lamda) + vim_score*lamda
            sig = vim_score[:t]
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #±êÇ©
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        np.save('result/ood.npy', pred)
        #fpr, tpr, threshold = roc_curve(list(gt), pred) #ÕæÊµ±êÇ©£¬Ô¤²â±êÇ©
        #rec_auc = auc(fpr, tpr)
        #print('lamda = '+ str(lamda) +', ood_auc = ' + str(rec_auc))

def test_scores_calc(dataloader, model, args, device):
    with torch.no_grad():
        model.eval() #¹Ì¶¨²ÎÊý
        pred = torch.zeros(0) #¿Õtensor

        for i, inputs in enumerate(dataloader): #iÎªÐòÁÐ
            input,t = inputs
            input = input.to(device)
            #input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
            feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, score_select_abnormal, score_select_normal,feats,logits = model(input) #logitsÎªscores
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            sig = scores[:t]
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy') #±êÇ©
        else:
            gt = np.load('list/gt-ucf.npy')
        
        pred = list(pred.cpu().detach().numpy()) #[T*batch_num]
        pred = np.repeat(np.array(pred), 16) #[T*batch_num*16]
        np.save('result/score.npy', pred)
        #fpr, tpr, threshold = roc_curve(list(gt), pred) #ÕæÊµ±êÇ©£¬Ô¤²â±êÇ©
        #rec_auc = auc(fpr, tpr)
        #print('lamda = '+ str(lamda) +', ood_auc = ' + str(rec_auc))
          
def real_test_ood(dataloader, model, args, device,lamda):
    with torch.no_grad():
        model.eval() #固定参数

        for i, input in enumerate(dataloader): #i为序列
            input = input.to(device)
            input = input.permute(0, 2, 1, 3) #[b,T,10,F] -> [b,10,T,F]
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, scores, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes,score_select_abn,score_select_nor,feats,logits = model(inputs=input) #logits为scores
            vim_score = ViM(feats,scores)
            scores = torch.squeeze(scores,2) #[b,T,1]->[b,T]
            scores = torch.mean(scores, 0) #[T]
            sig = scores*(1-lamda) + vim_score*lamda + 1.5
            print(sig)
            for i in range(len(sig)):
                print(sig[i])  
            #pred = torch.cat((pred, sig))
        

                   
