import numpy as np
import torch
from argparse import ArgumentParser
import os

from util.trainer import trainer
from util.tester import tester

torch.autograd.set_detect_anomaly(True)

def get_args():
    parser = ArgumentParser(description = "Hyperparameters")
    
    parser.add_argument('-dn', '--dataset-name', type = str, help = 'dataset name', dest = 'dataset_name', default = 'yahoo')    
    parser.add_argument('-mn', '--model-name', type = str, help = 'name of model', dest = 'model_name', default = 'wcpd')    
    
    parser.add_argument('-ms', '--manual-seed', type = int, help = 'manual seed', dest = 'manual_seed', default = 18) 
    
    #device
    parser.add_argument('-nw', '--num-workers', type = str, help = 'Number of workers', dest = 'num_workers', default = 0)
    parser.add_argument('-d', '--device', type = str, help = 'device', dest = 'device', default = 'cuda')
    parser.add_argument('-cn', '--cuda-num', type = int, help = 'cuda num', dest = 'cuda_num', default = 2) 
    
    #model
    parser.add_argument('-hd', '--hidden-dim', type=float, default=10,help='hidden dimension', dest = 'dim_h')
    parser.add_argument('-wd', '--wnd-dim', type=int,  default=25, help='window size (past and future)', dest = 'wnd_dim')
    
    #optim
    parser.add_argument('-ne', '--num-epochs', type = int, help = 'number of epochs', dest = 'num_epochs', default =  100)
    parser.add_argument('-lr', '--learning-rate', type = float, help = 'learning rate', dest = 'lr', default = 0.3)
    
    #data
    parser.add_argument('-bs', '--batch-size', type = int, help = 'batch size', dest = 'batch_size', default = 128)
    parser.add_argument('-bdd', '--base-dataset-dir', type = str, help = 'base dataset dir', dest = 'base_dataset_dir', default = './data/')
    parser.add_argument('-edd', '--exact-dataset-dir', type = str, help = 'exact dataset dir', dest = 'exact_dataset_dir', default = 'yahoo/yahoo-7.mat')
    parser.add_argument('--trn-ratio', type=float, default=0.5,help='how much data used for training', dest = 'r_train')
    parser.add_argument('--val-ratio', type=float, default=0.75,help='how much data used for validation', dest = 'r_val')
    
    #out
    parser.add_argument('-dod', '--default-output-dir', type = str, help = 'default output dir', dest = 'default_output_dir', default = './result')
    return parser.parse_args()

if __name__ == '__main__':
  
  args = get_args()  

  print(args.exact_dataset_dir) 
  
  if args.device =='cuda':
    torch.cuda.set_device(args.cuda_num)
  manual_seed = args.manual_seed
  np.random.seed(manual_seed)
  torch.manual_seed(manual_seed)

  os.makedirs(args.default_output_dir, exist_ok=True)

  trainer(args)
  tester(args)
  print('done')