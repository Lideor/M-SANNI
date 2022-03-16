# -*- coding: utf-8 -*-

import json

from DataRead.PredictorLoader import PredictorDataset
from Models.Classifier import Classifier
from Models.Predictor import Predictor
from Preprocess.preprocess import create_dataset
from DataRead.ClassifierLoader import ClassifierDataset
from pathlib import Path
from sklearn.metrics import precision_score, mean_squared_error, accuracy_score
from Models.Helpers import train, run_model
import torch
import numpy as np
import argparse

from Preprocess.const import *

torch.set_default_dtype(torch.float32)
# Fixme добавить полное описание ключей
def init_args():
    parser = argparse.ArgumentParser(description='M-SANNI')
    parser.add_argument('-s', "--size_subsequent", type=int,
                        action='store', dest='size_subsequent', help='size subsequent')
    parser.add_argument('-b', "--batch_size", type=int,
                        action='store', dest='batch_size', help='batch size')
    parser.add_argument('-d', "--dataset", type=lambda x: Path() / "Dataset" / x,
                        action='store', dest='dataset', help='dataset name')
    parser.add_argument('-c', "--count_snippet", type=int,
                        action='store', dest='count_snippet', help='count snippet')
    parser.add_argument('-g', default=False, action="store_true", dest='gpu', help='train on gpu')
    parser.add_argument('--epoch_predict', default=300,
                        action="store",type=int,
                        dest='epoch_pr', help='train on gpu')
    parser.add_argument('--epoch_classifier', default=10,type=int,
                        action="store", dest='epoch_cl', help='train on gpu')
    return parser.parse_args()


if __name__ == '__main__':

    parser_args = init_args()
    print(parser_args)

    # torch.set_default_dtype(torch.double)
    if parser_args.gpu and torch.cuda.is_available():
        CUDA = True
    else:
        CUDA = False
    print(parser_args.gpu)
    device = torch.device('cuda' if CUDA else 'cpu')
    run_model(size_subsequent=parser_args.size_subsequent,
              dataset=parser_args.dataset,
              epoch_cl=parser_args.epoch_cl,
              epoch_pr=parser_args.epoch_pr,
              count_snippet=parser_args.count_snippet,
              batch_size=parser_args.batch_size,
              device=device)
    #
    #
    # #create_dataset(size_subsequent, dataset, count_snippet)
    # cl_dataset = ClassifierDataset(dataset,
    #                                batch_size=batch_size,
    #                                size_subsequent=size_subsequent)
    # classifier = Classifier(size_subsequent=size_subsequent,
    #                         count_snippet=count_snippet,
    #                         dim=dim)
    # if torch.cuda.is_available():
    #     classifier.cuda()
    # loss = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=1.0e-3)
    # epochs = 20
    #
    #
    # def prec_score(predict, true):
    #     error = 0
    #     y_pred = predict.argmax(dim=1)
    #     prec = []
    #     acc = []
    #     for i in range(true.shape[1]):
    #         prec.append(precision_score(y_pred=y_pred[:, i].numpy(),
    #                                     y_true=true[:, i].numpy()))
    #         acc.append(accuracy_score(y_pred=y_pred[:, i].numpy(),
    #                                   y_true=true[:, i].numpy()))
    #
    #     return {"acc": min(acc), "prec": min(prec)}
    #
    #
    # _, classifier = train(model=classifier,
    #                       loader=cl_dataset.get_loader("train"),
    #                       loader_val=cl_dataset.get_loader("val"),
    #                       epochs_count=epochs,
    #                       optimizer=optimizer,
    #                       loss=loss,
    #                       device=device,
    #                       score_func=prec_score)
    # accuracy = 0
    # mse = 0
    # len_val = len(cl_dataset.get_loader("test"))
    # predict_true = []
    # predict_predict = []
    # for item in cl_dataset.get_loader("test"):
    #     x = item[0].to(device)
    #     y_pred = classifier.forward(x)
    #     y_true = item[1][:]
    #     # print(prec_score(y_pred, y_true))
    #     predict_true.append(y_true)
    #     predict_predict.append(y_pred.detach())
    # del cl_dataset
    # predict_true = torch.cat(predict_true)
    # predict_predict = torch.cat(predict_predict)
    # score_val = prec_score(predict_predict, predict_true)
    #
    # print(f"Test:{score_val}")
    #
    # # for item in pr_dataset.get_loader("train"):
    # #     x = item[0].to(device)
    # #     y_pred = classifier.forward(x)
    # #     y_true = item[1][1][:]
    # #     print(prec_score(y_pred, y_true))
    # # print(f"Test:{mse / len_val}")
    #
    # pr_dataset = PredictorDataset(dataset,
    #                               batch_size=batch_size,
    #                               size_subsequent=size_subsequent)
    # predictor = Predictor(300, 2,
    #                       classifier=classifier,
    #                       snippet_list=pr_dataset.original_snippet,
    #                       dim=dim)
    #
    # if torch.cuda.is_available():
    #     predictor.cuda()
    # epochs = 300
    # loss = torch.nn.MSELoss()
    #
    # optimizer = torch.optim.Adam(predictor.parameters(), lr=1.0e-4)
    # h0 = predictor.init_hidden(pr_dataset.batch_size)
    # history = train(model=predictor,
    #                 loader=pr_dataset.get_loader("train"),
    #                 loader_val=pr_dataset.get_loader("val"),
    #                 epochs_count=300,
    #                 optimizer=optimizer,
    #                 loss=loss,
    #                 device=device,
    #                 score_func=
    #                 lambda y_pred,
    #                        y_true:
    #                 mean_squared_error(y_pred=y_pred.detach().numpy(), y_true=y_true.detach().numpy()),
    #                 )
    # # history = {"train": [], "val": []}
    # # for epoch in range(epochs):
    # #     mse_train = 0
    # #     len_val = len(pr_dataset.get_loader("train"))
    # #
    # #     for item in pr_dataset.get_loader("train"):
    # #         optimizer.zero_grad()
    # #         x = item[0].to(device)
    # #
    # #         y_pred = predictor.forward(x)
    # #         y_true = item[1]
    # #         loss_value = loss(y_pred, y_true)
    # #         loss_value.backward()
    # #         optimizer.step()
    # #         print(y_pred.shape,y_true.shape)
    # #         mse_train += mean_squared_error(y_pred=y_pred.detach().numpy(), y_true=y_true.detach().numpy())
    # #     mse_train /= len_val
    # #     history["train"].append(mse_train)
    # #     len_val = len(pr_dataset.get_loader("val"))
    # #     mse = 0
    # #     for item in pr_dataset.get_loader("val"):
    # #         x = item[0].to(device)
    # #         y_pred = predictor.forward(x)
    # #         y_true = item[1][:, None]
    # #         mse += mean_squared_error(y_pred=y_pred.detach().numpy(), y_true=y_true.detach().numpy())
    # #     print(f"{epoch + 1}/{epochs} Train:{mse_train} val:{mse / len_val}")
    # #     history["val"].append(mse)
    # mse = 0
    # len_val = len(pr_dataset.get_loader("test"))
    # for item in pr_dataset.get_loader("test"):
    #     x = item[0].to(device)
    #     y_pred = predictor.forward(x)
    #     y_true = item[1][:]
    #     mse += mean_squared_error(y_pred=y_pred.detach().numpy(), y_true=y_true.detach().numpy())
    # print(f"Test:{mse / len_val}")
    # json.dump(history, open(dataset / "history.json", "w"))
