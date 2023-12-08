import time
import numpy as np
import torch
import torch.optim as optim
import sklearn.metrics
from torch.autograd import Variable
import os


from util.dataloader import get_loader
from util.util import frozen_params,free_params
from util.metrics import batch_wasserstein_distance


def tester(args):


  train_loader,valid_loader,test_loader = get_loader(args)

  if os.path.exists(args.default_output_dir+'/best_netD_'+args.model_name+'_'+args.dataset_name+'.pth'):
    netD = torch.load(args.default_output_dir+'/best_netD_'+args.model_name+'_'+args.dataset_name+'.pth')
    print("Checkpoints correctly loaded: ", 'best_netD_'+args.model_name+'_'+args.dataset_name+'.pth')
      
  L = []
  Y_pred = []
  for X_p, X_f, label in test_loader: 

    X_p = X_p.to(args.device)#B*10*1
    X_f = X_f.to(args.device)
    L.append(label)

    X_p_enc, X_p_dec = netD(X_p)
    X_f_enc, X_f_dec = netD(X_f)
    
    w_batch = []
    for i in range(X_p_enc.shape[2]):
        w_batch.append(batch_wasserstein_distance(X_p_enc[:,:,i],X_f_enc[:,:,i]).unsqueeze(1))
    Y_pred_batch = torch.mean(torch.cat(w_batch,dim=1),dim=1) 
    Y_pred.append(Y_pred_batch)
  Y_pred = torch.cat(Y_pred)

  fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(torch.cat(L).squeeze(1).cpu().detach().numpy(), Y_pred.cpu().detach().numpy())
  auc = sklearn.metrics.auc(fp_list, tp_list)
    
  print(f'test:auc -> {"{:.3f}".format(auc*100)} %')




