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
from torch import tensor, transpose
from tqdm import tqdm
torch.set_default_dtype(torch.double)

class ClassifierLoader(Dataset):
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassifierDataset:
    def __init__(self, data_dir: Path,
                 shuffle=False,
                 random=2366,
                 batch_size=32,
                 size_subsequent=100,
                 test_size=0.25,
                 log=True):

        self.data_dir = data_dir
        self.shuffle = shuffle
        self.random = random
        self.batch_size = batch_size
        self.windows = json.load(open(self.data_dir / CURRENT_PARAMS_FILE_NAME))["size_subsequent"]
        snippet_list_arr = []
        snippet_count = []
        print("загрузка сниппетов")
        for path_snippet in self.data_dir.glob("*.csv.gz"):
            snippet = pd.read_csv(path_snippet,
                                  compression='gzip',

                                  usecols=["key", "neighbors_index"],
                                  converters={"neighbors_index": json.loads})
            snippet_count.append(snippet.shape[0])
            snippet_list_arr.append(snippet)

        test_y = []
        for item in snippet_list_arr:
            buf = []
            for ind, row in item.iterrows():
                x = torch.zeros(len(row.neighbors_index), 2)
                x[:, 0] = tensor(row.neighbors_index)
                x[:, 1] = row["key"]
                buf.append(x)
            x = np.vstack(buf)
            x = tensor(sorted(x, key=lambda x: x[0]), dtype=torch.long)
            test_y.append(x[:, None, 1])

        y = torch.cat(test_y, 1)

        data_norm = np.loadtxt(self.data_dir / NORM_DATA_FILE_NAME)
        if len(data_norm.shape) < 2:
            data_norm = data_norm.reshape(-1, 1)
        X = []
        if log:
            pbar = tqdm(range(0, len(data_norm) - size_subsequent + 1))
        else:
            pbar = range(0, len(data_norm) - size_subsequent + 1)
        for indx in pbar:
            X.append(data_norm[indx:indx + size_subsequent - 1, :].tolist())

        del snippet_list_arr
        X = tensor(X).transpose(1, 2)
        y = y[:X.shape[0]]
        # X = snippet.iloc[0].neighbors
        #
        # X = np.array(X)[[len(X[i]) == self.windows for i in range(len(X))]]
        # X = np.stack(X)[:, :-1]
        # y = np.full(len(X), snippet.iloc[0].key)
        # for i, item in snippet.iloc[1:].iterrows():
        #     neighbors = item.neighbors
        #     neighbors = np.array(neighbors)[[len(neighbors[i]) == self.windows for i in range(len(neighbors))]]
        #     neighbors = np.stack(neighbors)[:, :-1]
        #     keys = np.full(len(neighbors), item.key)
        #     X = np.vstack([X, neighbors])
        #     y = np.hstack([y, keys])
        self.dataset = {}
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random)
        del X
        del y
        self.dataset["test"] = [X_test, y_test]
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          test_size=test_size,
                                                          random_state=random)
        self.dataset["val"] = [X_val, y_val]
        self.dataset["train"] = [X_train, y_train]

    def get_loader(self, type_dataset):
        return DataLoader(ClassifierLoader(self.dataset[type_dataset][0], self.dataset[type_dataset][1]),
                          batch_size=self.batch_size)
