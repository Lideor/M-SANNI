# -*- coding: utf-8 -*-
import json

import torch as th
from tqdm import tqdm
from DataRead.PredictorLoader import PredictorDataset
from Models.Classifier import Classifier
from Models.Predictor import Predictor
from Preprocess.const import DATA_FILE_NAME
from Preprocess.preprocess import create_dataset
from DataRead.ClassifierLoader import ClassifierDataset
from pathlib import Path
from sklearn.metrics import precision_score, mean_squared_error, accuracy_score
import torch
import numpy as np


def run_epoch(models,
              optimizer,
              loss,
              x,
              device,
              y_true):
    optimizer.zero_grad()
    x = x.to(device)
    y_true = y_true.to(device)
    y_pred = models.forward(x)
    loss_value = loss(y_pred, y_true)
    loss_value.backward()
    optimizer.step()
    return y_pred


def train(model,
          loader,
          loader_val,
          epochs_count,
          optimizer,
          loss,
          device,
          score_func,
          bar=False,
          log=True):
    history = {"train": [],
               "val": []}
    model.train()
    for epoch in range(epochs_count):
        predict_true = []
        predict_predict = []
        if log and bar:
            pbar = tqdm(loader, desc=f"{epoch + 1}/{epochs_count}")
        else:
            pbar = loader
        for x, y_true in pbar:
            y_true
            y_pred = run_epoch(model,
                               optimizer,
                               loss,
                               x, device, y_true)
            predict_true.append(y_true)
            predict_predict.append(y_pred)
        predict_true = th.cat(predict_true)
        predict_predict = th.cat(predict_predict)
        score_train = score_func(predict_predict, predict_true)
        del predict_predict, predict_true
        predict_true = []
        predict_predict = []

        for x, y_true in loader_val:
            y_true.to(device)
            optimizer.zero_grad()
            x = x.to(device).detach()
            y_pred = model.forward(x)
            predict_true.append(y_true)
            predict_predict.append(y_pred.detach())

        predict_true = th.cat(predict_true)
        predict_predict = th.cat(predict_predict)
        score_val = score_func(predict_predict, predict_true)
        if log:
            print(f"\r Train:{score_train} Val:{score_val}")
        history["train"].append(score_train)
        history["val"].append(score_val)

    return history, model


# FIXME переделать под локоть
def run_model(size_subsequent, dataset, count_snippet, batch_size, device,
              epoch_cl=1, epoch_pr=300):
    p = Path(dataset / DATA_FILE_NAME)
    dim = np.loadtxt(p).shape[1]
    # create_dataset(size_subsequent, dataset, count_snippet)

    cl_dataset = ClassifierDataset(dataset,
                                   batch_size=batch_size,
                                   size_subsequent=size_subsequent,
                                   device=device)
    classifier = Classifier(size_subsequent=size_subsequent,
                            count_snippet=count_snippet,
                            dim=dim)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1.0e-3)

    def prec_score(predict, true):
        error = 0
        y_pred = predict.argmax(dim=1)
        prec = []
        acc = []
        for i in range(true.shape[1]):
            prec.append(precision_score(y_pred=y_pred[:, i].cpu().numpy(),
                                        y_true=true[:, i].cpu().numpy()))
            acc.append(accuracy_score(y_pred=y_pred[:, i].cpu().numpy(),
                                      y_true=true[:, i].cpu().numpy()))

        return {"acc": min(acc), "prec": min(prec)}

    classifier.to(device)
    _, classifier = train(model=classifier,
                          loader=cl_dataset.get_loader("train"),
                          loader_val=cl_dataset.get_loader("val"),
                          epochs_count=epoch_cl,
                          optimizer=optimizer,
                          loss=loss,
                          device=device,
                          score_func=prec_score)
    accuracy = 0
    mse = 0
    len_val = len(cl_dataset.get_loader("test"))
    predict_true = []
    predict_predict = []
    for item in cl_dataset.get_loader("test"):
        x = item[0].to(device)
        y_pred = classifier.forward(x).cpu()
        y_true = item[1][:]
        # print(prec_score(y_pred, y_true))
        predict_true.append(y_true)
        predict_predict.append(y_pred.detach())
    del cl_dataset
    predict_true = torch.cat(predict_true)
    predict_predict = torch.cat(predict_predict)
    score_val = prec_score(predict_predict, predict_true)

    print(f"Test:{score_val}")
    pr_dataset = PredictorDataset(dataset,
                                  batch_size=batch_size,
                                  size_subsequent=size_subsequent,
                                  device=device)
    predictor = Predictor(size_subsequent, count_snippet,
                          classifier=classifier,
                          snippet_list=pr_dataset.original_snippet,
                          dim=dim,
                          device=device)

    predictor = predictor.to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1.0e-4)
    history, _ = train(model=predictor,
                       loader=pr_dataset.get_loader("train"),
                       loader_val=pr_dataset.get_loader("val"),
                       epochs_count=epoch_pr,
                       optimizer=optimizer,
                       loss=loss,
                       device=device,
                       bar=True,
                       score_func=
                       lambda y_pred,
                              y_true:
                       mean_squared_error(y_pred=y_pred.cpu().detach(), y_true=y_true.cpu().detach()),
                       )
    mse = 0
    len_val = len(pr_dataset.get_loader("test"))
    for item in pr_dataset.get_loader("test"):
        x = item[0].to(device)
        y_pred = predictor.forward(x)
        y_true = item[1][:]
        mse += mean_squared_error(y_pred=y_pred.cpu().detach(), y_true=y_true.cpu().detach())
    print(f"Test:{mse / len_val}")
    print(history)
    json.dump(str(history), open(dataset / "history.json", "w"))
