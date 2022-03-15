# -*- coding: utf-8 -*-

from torch import nn
import torch


class Classifier(nn.Module):
    def __init__(self, size_subsequent: int, count_snippet: int, dim: int):
        super(Classifier, self).__init__()
        self.size_subsequent = size_subsequent
        self.cnns_layers = []
        self.dim = dim
        self.hidden_size = 32
        self.count_snippet = count_snippet
        in_channels = dim
        self.out_channels = 256

        for i in range(3):
            self.out_channels = self.out_channels // 2
            self.cnns_layers.append({"conv": nn.Conv1d(in_channels=in_channels,
                                                       out_channels=self.out_channels,
                                                       padding=2,
                                                       kernel_size=(5,)),
                                     "aver": nn.MaxPool1d(2)})
            in_channels = self.out_channels

        self.gru = nn.GRU(input_size=self.out_channels,
                          batch_first=True,
                          hidden_size=self.hidden_size)
        # self.last = nn.Linear(self.hidden_size, count_snippet * dim)
        self.linear = nn.Linear(self.out_channels * 37, count_snippet * dim)
        self.last = nn.Linear(self.out_channels * 37, count_snippet * dim)

    def forward(self, x):
        print(x.shape)
        for layer in self.cnns_layers:
            x = layer["conv"](x)
            print(x.shape)
            x = nn.ReLU()(x)
            x = layer["aver"](x)
            print(x.shape)
        # x = x.transpose(1, 2)
        # x, _ = self.gru(x)
        # x = nn.ReLU()(x)
        # x = x[:, -1, :]
        x = x.reshape(-1, self.out_channels * 33)
        print(x.shape)
        x = self.last(x)
        x = x.reshape(-1, self.count_snippet, self.dim)
        print(x.shape)
        x = nn.Softmax(dim=1)(x)
        return x

    def get_loss(self, predict, true):
        print(predict[:, :, 0].shape, true.shape)
        loss_arr = torch.nn.CrossEntropyLoss()(predict[:, :, 0], true[:, 0])
        for i in range(1, self.dim):
            loss_arr += torch.nn.CrossEntropyLoss()(predict[:, :, i], true[:, i])
        return loss_arr
