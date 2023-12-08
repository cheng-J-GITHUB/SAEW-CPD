import os

import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np

   
class Dataset(BaseDataset):
    """Read data    
    """
    
    def __init__(
            self, 
            root,
            mode,
            args,
        ):     
        self.root = root
        # self.ids = os.listdir(self.root)
        self.ids = args.exact_dataset_dir

        self.Pre = []
        self.Post = []
        self.label = []
        # for i in self.ids:
        # dir = self.root +i
        dir1 = self.root +self.ids
        data = sio.loadmat(dir1)
        data['Y'] = data['Y'].astype(np.float32)#.reshape(data['Y'].shape[1],-1)
        for j in range(data['Y'].shape[0]-args.wnd_dim-10):
          self.Pre.append(data['Y'][j:j+args.wnd_dim,:])
          self.Post.append(data['Y'][j+args.wnd_dim:j+args.wnd_dim+10,:])
          self.label.append(data['L'][j+args.wnd_dim])
        if mode == 'train':
          self.Pre_d = self.Pre[:int(len(self.Pre)*args.r_train)]
          self.Post_d = self.Post[:int(len(self.Pre)*args.r_train)]
          self.label_d = self.label[:int(len(self.Pre)*args.r_train)]
        elif mode == 'val':
          self.Pre_d = self.Pre[int(len(self.Pre)*args.r_train):int(len(self.Pre)*args.r_val)]
          self.Post_d = self.Post[int(len(self.Pre)*args.r_train):int(len(self.Pre)*args.r_val)]
          self.label_d = self.label[int(len(self.Pre)*args.r_train):int(len(self.Pre)*args.r_val)]
        else:
          self.Pre_d = self.Pre[int(len(self.Pre)*args.r_val):]
          self.Post_d = self.Post[int(len(self.Pre)*args.r_val):]
          self.label_d = self.label[int(len(self.Pre)*args.r_val):]
             
        self.dataset_name = args.dataset_name
        
    def __getitem__(self, i):
            
        return self.Pre[i], self.Post[i], self.label[i]

        
    def __len__(self):

      return len(self.Pre_d)-1


def get_loader(args):
  
    x_dir = args.base_dataset_dir+'/'#+args.dataset_name

    train_dataset = Dataset(x_dir, 'train', args)
    valid_dataset = Dataset(x_dir, 'val', args)
    test_dataset = Dataset(x_dir, 'test', args)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=args.num_workers)
    
    return train_loader,valid_loader,test_loader


    