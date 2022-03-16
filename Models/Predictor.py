# -*- coding: utf-8 -*-
import multiprocess
import pandas as pd
from torch import nn
import torch
import time
from joblib import Parallel, delayed
from typing import List

from Models.Classifier import Classifier


class Predictor(nn.Module):
    def __init__(self, size_subsequent: int,
                 count_snippet: int,
                 dim: int,
                 classifier: Classifier,
                 snippet_list,
                 device,
                 hidden_dim=128,
                 num_layers=1,
                 ):
        super(Predictor, self).__init__()
        self.size_subsequent = size_subsequent
        self.classifier = classifier.eval()
        self.device = device

        print(snippet_list)
        self.snippet_list = torch.tensor(snippet_list,
                                         device=self.device)
        print(self.snippet_list.shape)
        self.dim = dim
        self.num_layers = 1
        self.hidden_dim = 128
        self.gru = nn.GRU(input_size=self.dim * 2,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_layers,
                          batch_first=True)
        self.relu = nn.ReLU()
        self.fs1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.relu1 = nn.ReLU()
        self.fs2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.relu2 = nn.ReLU()
        self.fs3 = nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8)
        self.relu3 = nn.ReLU()
        self.last = nn.Linear(self.hidden_dim // 8 + dim, dim)
        self.relu_last = nn.ReLU()

    def forward(self, x):
        # full_time = time.time()
        # start_time = time.time()
        with torch.no_grad():
            snippet = self.classifier(x).argmax(dim=1).cpu()
        # torch.cuda.synchronize()
        # print("1--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()

        snip = self.snippet_tensor(snippet)
        # print("1.1--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        # snip = snip.to(self.device)

        last = snip[:, :, -1]
        snip = snip[:, :, :-1]
        # print("1.2--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        x = torch.cat((x, snip), dim=1)
        # print("2--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()

        x = x.transpose(1, 2)
        x, h = self.gru(x)
        #

        x = x[:, -1, :]
        # x = nn.LeakyReLU()(x)
        x = self.fs1(x)
        x = nn.LeakyReLU()(x)
        x = self.fs2(x)
        x = nn.LeakyReLU()(x)
        x = self.fs3(x)
        x = nn.LeakyReLU()(x)
        x = torch.cat((x, last), dim=1)
        x = self.last(x)
        x = nn.LeakyReLU()(x)
        # print("3--- %s seconds ---" % (time.time() - start_time))
        # print("3--- %s seconds ---" % (time.time() - full_time))
        return x

    def snippet_tensor(self, snippet):
        arr = []
        # snippet=snippet.cpu()
        # snippet = snippet.cpu()
        # print("переход %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        # for i in range(self.dim):
        #     snippet[:,i].apply_(lambda x: self.test_func(x, i))
        #
        # print("1.1--- %s seconds ---" % (time.time() - start_time))
        # snip = pd.DataFrame(snippet.cpu()).apply(self.get_snippet, axis=1)
        #
        # # results = Parallel(n_jobs=2)(delayed(self.get_snippet)(i.tolist()) for i in snip)
        # # pool_obj = multiprocess.Pool()
        # # answer = pool_obj.map(lambda x: self.get_snippet(x.tolist()),
        # #                       range(0, 5))
        # # print(snip.shape)
        # print("1.1--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        result = torch.empty(snippet.shape[0],
                             self.dim,
                             self.size_subsequent,
                             device=self.device)
        #
        # print("переход %s seconds ---" % (time.time() - start_time))
        #
        # futures: List[torch.jit.Future[torch.Tensor]] = []
        # start_time = time.time()
        #
        # class Snippet(torch.nn.Module):
        #     def forward(forw, ids, item):
        #         return self.snippet_list[ids, item]
        #
        # class AddMod(torch.nn.Module):
        #     def forward(forw, number):
        #
        #         arr = torch.empty(self.dim,
        #                           self.size_subsequent,
        #                           device=self.device)
        #
        #
        #         for ids, item in enumerate(number):
        #
        #             arr[ids,:] = Snippet()(ids, item)
        #         print(arr.device,number.device)
        #         return arr
        #
        # for batch_number in range(snippet.shape[0]):
        #     futures.append(torch.jit.fork(AddMod(), snippet[batch_number]))
        #
        # for batch_number, future in enumerate(futures):
        #     result[batch_number, :, :] = torch.jit.wait(future)
        # print(result.device)
        #
        # print("парал--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        for batch_number in range(snippet.shape[0]):
            for ids in range(self.dim):
                result[batch_number, ids, :] = self.snippet_list[ids, snippet[batch_number, ids]]
        # print("итерация--- %s seconds ---" % (time.time() - start_time))
        return result

    def test_func(self, x):
        return x

    def get_snippet(self, number):
        arr = torch.empty(self.dim,
                          self.size_subsequent,
                          device=self.device)
        for ids, item in enumerate(number):
            arr[ids, :] = self.snippet_list[ids, item]
        return arr

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers,
                            batch_size,
                            self.hidden_dim).zero_().to(torch.device('cpu'))
        return hidden
