import time
import numpy as np
import torch
import torch.optim as optim
import sklearn.metrics
from torch.autograd import Variable

from util.dataloader import get_loader
from util.metrics import batch_wasserstein_distance
from models.model import NetD,NetG


def trainer(args):

  
  # load dataset
  train_loader,valid_loader,test_loader = get_loader(args)

  #define/load model
  netD = NetD(args).to(args.device)
  netG = NetG(args).to(args.device)

  #define optimizer
  optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr = args.lr)
  optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr = args.lr)
  scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size = 100, gamma = 0.6,)
  scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size = 100, gamma = 0.6,)

  
  start = time.time()
  bestmetric = 0 
  
  one = torch.tensor(1, dtype=torch.float)
  #start train
  for epoch in range(args.num_epochs):

    total_loss_D = []
    total_loss_real = []
    total_loss_rf = []
    for X_p, X_f, label in train_loader:
      
      netD.train()  
      
      X_p = X_p.to(args.device)#B*10*C
      X_f = X_f.to(args.device)
      label = label.squeeze(1).to(args.device)
      
      # real data
      X_p_enc, X_p_dec = netD(X_p)
      X_f_enc, X_f_dec = netD(X_f)

      # fake data
      noise = torch.cuda.FloatTensor(1, X_p.shape[0], args.dim_h).normal_(0, 1)
      Y_f = netG(X_p, X_f, noise)
      Y_f_enc, Y_f_dec = netD(Y_f)

      #reconstruction loss
      w_batch = []
      for i in range(X_f_enc.shape[2]):
          w_batch.append(batch_wasserstein_distance(X_p_enc[:,:,i],X_f_enc[:,:,i]).unsqueeze(1))
      w_batch = torch.mean(torch.cat(w_batch,dim=1),dim=1)
      # w_batch = (w_batch-torch.min(w_batch))/(torch.max(w_batch)-torch.min(w_batch))
      mid_w = torch.mul(label,w_batch)
      lower_limit = torch.min(torch.where(mid_w==0,torch.max(w_batch),w_batch))
      weight = torch.where(w_batch>lower_limit,label+1,label)+1
      label = torch.where(label == 1, torch.max(w_batch), torch.min(w_batch))
      real = torch.mul(weight,torch.abs(label-w_batch))
      
      # reconstruction loss
      real_L2_loss = torch.mean((X_f - X_f_dec)**2)
      #real_L2_loss = torch.mean((X_p - X_p_dec)**2)
      fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2)
      #fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2) * 0.0

      netD.zero_grad()
      lossD = real.mean() - 0.001*(real_L2_loss + fake_L2_loss)#- 0.001 * 
      lossD.backward(one)
      optimizer_D.step()
      total_loss_real.append(real.mean())      
      total_loss_rf.append(real_L2_loss + fake_L2_loss)
      total_loss_D.append(lossD)
    
    scheduler_D.step()
    
    print('[%3d] real %.4e other %.4e lossD %.4e '% (epoch, torch.mean(torch.stack(total_loss_real)),  torch.mean(torch.stack(total_loss_rf)), torch.mean(torch.stack(total_loss_D))))
                                                                  
    with torch.no_grad():
      netD.eval()

      L = []
      Y_pred = []
      for X_p, X_f, label in valid_loader: 

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

      fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(torch.cat(L).squeeze(1).cpu(), Y_pred.cpu())
      auc = sklearn.metrics.auc(fp_list, tp_list)
        
      print(f'Val:auc -> {"{:.3f}".format(auc*100)} %')

      if auc > bestmetric:
        bestmetric = auc
        torch.save(netD, args.default_output_dir+'/best_netD_'+args.model_name+'_'+args.dataset_name+'.pth')
        
        print('Best model saved!')

  end = time.time()
  print('Training completed. Program processed ', (end - start)/3600, 'h')
  print(f'Best metrics: auc score -> {bestmetric*100} %')



