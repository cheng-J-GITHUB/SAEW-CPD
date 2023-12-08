from __future__ import print_function
import pickle as pickle
import torch
import torch.nn as nn


class NetG(nn.Module):
    def __init__(self, args):
        super(NetG, self).__init__()
        if args.dataset_name in ['beedance','hasc']:
            self.var_dim = 3
        elif args.dataset_name in ['yahoo','fishkiller']:
            self.var_dim = 1
        else:
            self.var_dim = 1
        self.RNN_hid_dim = args.dim_h

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
        self.fc_layer = nn.Linear(self.RNN_hid_dim, self.var_dim)


    def forward(self, X_p, X_f, noise):
        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X):
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft


class NetD(nn.Module):
    def __init__(self, args):
        super(NetD, self).__init__()

        if args.dataset_name in ['beedance','hasc']:
            self.var_dim = 3
        elif args.dataset_name in ['yahoo','fishkiller']:
            self.var_dim = 1
        else:
            self.var_dim = 1
        self.RNN_hid_dim = args.dim_h

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.var_dim, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec