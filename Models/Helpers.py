# -*- coding: utf-8 -*-
import json
import copy
import torch as th
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from DataRead.PredictorLoader import PredictorDataset
from Models.Classifier import Classifier
from Models.Predictors.Hard import Hard
from Models.Predictors.Predictor import Predictor
from Models.Predictors.OnlyCnns import OnlyCnns
from Preprocess.const import DATA_FILE_NAME, CURRENT_PARAMS_FILE_NAME
from Preprocess.preprocess import create_dataset, get_score
from DataRead.ClassifierLoader import ClassifierDataset
from pathlib import Path
from sklearn.metrics import precision_score, mean_squared_error, accuracy_score, recall_score, f1_score
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
    with torch.set_grad_enabled(True):
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
          loader_test=None,
          sh=False,
          early_stopping_patience=50,
          bar=False,
          log=True):
    history = {"train": [],
               "val": []}
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.75)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.95, verbose=True)

    for epoch in range(epochs_count):
        model.train()
        predict_true = []
        predict_predict = []
        if log and bar:
            pbar = tqdm(loader, desc=f"{epoch + 1}/{epochs_count}")
        else:
            pbar = loader
        for x, y_true in pbar:
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
        model.eval()
        with torch.no_grad():
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
            print(f"\r epoch {epoch} - Train:{score_train} Val:{score_val}")
        val_loss = loss(predict_predict, predict_true)
        if val_loss < best_val_loss:
            best_epoch_i = epoch
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            print('new best model')
        # elif epoch - best_epoch_i > early_stopping_patience:
        #     print(f'The model has not improved over the last {early_stopping_patience} epochs, stop training')
        #     break
        if sh:
            scheduler.step(val_loss)
        # if epoch % 100 == 0 and loader_test is not None:
        #     predict_true = []
        #     predict_predict = []
        #     with torch.set_grad_enabled(False):
        #         for x, y_true in loader_test:
        #             y_true.to(device)
        #             optimizer.zero_grad()
        #             x = x.to(device).detach()
        #             y_pred = model.forward(x)
        #             predict_true.append(y_true)
        #             predict_predict.append(y_pred.detach())
        #         predict_true = th.cat(predict_true)
        #         predict_predict = th.cat(predict_predict)
        #         score_test = get_score(predict_true.cpu().detach(),
        #                               predict_predict.cpu().detach())
        #         if log:
        #             print(f"\r Test:{score_test}")

        history["train"].append(score_train)
        history["val"].append(score_val)
        # scheduler.step()

    return history, best_model


# FIXME переделать под локоть
def run_model(size_subsequent,
              dataset,
              count_snippet,
              batch_size,
              device,
              num_layers=1,
              hidden=128,
              sh=False,
              bar=True,
              model="original",
              epoch_cl=1, epoch_pr=300):
    models = {
        "original": Predictor,
        "only_cnns": OnlyCnns,
        "hard": Hard
    }
    assert model in models.keys()

    p = Path(dataset / DATA_FILE_NAME)
    dim = np.loadtxt(p)
    if len(dim.shape) > 1:
        dim = dim.shape[1]
    else:
        dim = 1
    p = Path(dataset / CURRENT_PARAMS_FILE_NAME)
    current_params = json.load(open(p, "rb+"))
    print(current_params)
    if size_subsequent != current_params['size_subsequent']:
        count_snippet = create_dataset(size_subsequent, dataset, count_snippet)
    else:
        count_snippet = current_params['snippet_count']
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
        rec = []
        f1 = []
        for i in range(true.shape[1]):
            prec.append(precision_score(y_pred=y_pred[:, i].cpu().numpy(),
                                        y_true=true[:, i].cpu().numpy(), average='weighted'))
            acc.append(accuracy_score(y_pred=y_pred[:, i].cpu().numpy(),
                                      y_true=true[:, i].cpu().numpy()))
            rec.append(recall_score(y_pred=y_pred[:, i].cpu().numpy(),
                                    y_true=true[:, i].cpu().numpy(), average='weighted'))
            f1.append(f1_score(y_pred=y_pred[:, i].cpu().numpy(),
                               y_true=true[:, i].cpu().numpy(), average='weighted'))
        return {"acc": np.median(acc),
                "prec": np.median(prec),
                "rec": np.median(rec),
                "f1": np.median(f1)}

    classifier.to(device)

    _, classifier = train(model=classifier,
                          loader=cl_dataset.get_loader("train"),
                          loader_val=cl_dataset.get_loader("val"),
                          epochs_count=epoch_cl,
                          optimizer=optimizer,
                          bar=bar,
                          loss=loss,
                          device=device,
                          score_func=prec_score)
    accuracy = 0
    mse = 0
    len_val = len(cl_dataset.get_loader("test"))
    predict_true = []
    predict_predict = []
    classifier.eval()
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

    predictor = models[model](size_subsequent, count_snippet,
                              num_layers=num_layers,
                              classifier=classifier,
                              snippet_list=pr_dataset.original_snippet,
                              hidden_dim=hidden,
                              dim=dim,
                              device=device)

    predictor = predictor.to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1.0e-3, amsgrad=True)
    history, predictor = train(model=predictor,
                               loader=pr_dataset.get_loader("train"),
                               loader_val=pr_dataset.get_loader("val"),
                               loader_test=pr_dataset.get_loader("test"),
                               epochs_count=epoch_pr,
                               optimizer=optimizer,
                               loss=loss,
                               device=device,
                               bar=bar,
                               sh=sh,
                               score_func=
                               lambda y_pred,
                                      y_true:
                               mean_squared_error(y_pred=y_pred.cpu().detach(), y_true=y_true.cpu().detach()),
                               )
    len_val = len(pr_dataset.get_loader("test"))
    mse = np.zeros(dim)
    predictor.eval()

    predict_true = []
    predict_predict = []
    with torch.no_grad():
        for x, y_true in pr_dataset.get_loader("test"):
            y_true.to(device)
            optimizer.zero_grad()
            x = x.to(device).detach()
            y_pred = predictor.forward(x)
            predict_true.append(y_true.cpu().detach())
            predict_predict.append(y_pred.cpu().detach())

    predict_true = th.cat(predict_true)
    predict_predict = th.cat(predict_predict)
    score_val = get_score(predict_predict, predict_true)

    print(f"Test:{score_val}")
    json.dump(str(history), open(dataset / f"{model}_{size_subsequent}_{count_snippet}_sh={sh}_history.json", "w"))
    # json.dump(str({mse.tolist() / len_val}), open(dataset / f"{model}_{size_subsequent}_{count_snippet}_test.json", "w"))
    json.dump(str(score_val),
              open(dataset / f"{model}_{size_subsequent}_{count_snippet}_sh={sh}_all.json", "w"))
