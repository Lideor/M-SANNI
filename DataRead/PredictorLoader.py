# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import json
from Preprocess.const import *
from torch.utils.data import DataLoader
from torch import tensor


class PredictorLoader(Dataset):
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PredictorDataset:
    def __init__(self, data_dir: Path, size_subsequent,
                 shuffle=False, random=2366, batch_size=32, test_size=0.25):

        self.data_dir = data_dir
        self.shuffle = shuffle
        self.random = random
        self.size_subsequent = size_subsequent
        self.batch_size = batch_size

        self.data = np.loadtxt(data_dir / NORM_DATA_FILE_NAME)
        if len(self.data.shape) < 2:
            self.data = self.data.reshape(-1, 1)
        self.snippet_list_arr = []
        self.snippet_count = []
        self.original_snippet = []
        for path_snippet in self.data_dir.glob("*.csv.gz"):
            snippet = pd.read_csv(path_snippet,
                                  usecols=["key", "snippet", "neighbors_index"],
                                  compression='gzip',
                                  converters={"neighbors_index": json.loads,
                                              "snippet": json.loads})
            self.snippet_count.append(snippet.shape[0])
            self.snippet_list_arr.append(snippet)
            self.original_snippet.append(snippet.snippet.values)
        X = []
        for indx in range(0, len(self.data) - size_subsequent):
            X.append(self.data[indx:indx + size_subsequent, :].tolist())
        X = torch.Tensor(X)
        y = X[:, -1]
        X = X[:, :-1].transpose(1, 2)
        self.dataset = {}
        X_train = X[:y.shape[0] // 2]
        X_val = X[y.shape[0] // 2:2 * y.shape[0] // 3]
        X_test = X[2 * y.shape[0] // 3:]
        y_train = y[:y.shape[0] // 2]
        y_val = y[y.shape[0] // 2:2 * y.shape[0] // 3]
        y_test = y[2 * y.shape[0] // 3:]
        # X_train, X_test, y_train, y_test = train_test_split(X,
        #                                                     y,
        #                                                     test_size=test_size,
        #                                                     shuffle=False,
        #                                                     random_state=random)
        self.dataset["test"] = [X_test, y_test]
        # X_train, X_val, y_train, y_val = train_test_split(X_train,
        #                                                   y_train,
        #                                                   test_size=test_size,
        #                                                   shuffle=False,
        #                                                   random_state=random)
        self.dataset["val"] = [X_val, y_val]
        self.dataset["train"] = [X_train, y_train]

    def get_loader(self, type_dataset):
        return DataLoader(PredictorLoader(
            self.dataset[type_dataset][0],
            self.dataset[type_dataset][1],
            batch_size=self.batch_size))
