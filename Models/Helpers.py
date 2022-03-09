import numpy as np
import torch as th
from tqdm import tqdm


def run_epoch(models,
              optimizer,
              loss,
              x,
              device,
              y_true,
              h=None):
    optimizer.zero_grad()
    x = x.to(device)
    if h is None:
        y_pred = models.forward(x)
    else:
        y_pred, h = models.forward(x, h)
    loss_value = loss(y_pred, y_true)
    loss_value.backward()
    optimizer.step()
    return y_pred, h


def train(model,
          loader,
          loader_val,
          epochs_count,
          optimizer,
          loss,
          device,
          score_func,
          h=None,
          log=True):
    history = {"train": [],
               "val": []}
    model.train()
    for epoch in range(epochs_count):
        predict_true = []
        predict_predict = []
        if log:
            pbar = tqdm(loader, desc=f"{epoch + 1}/{epochs_count}")
        else:
            pbar = loader
        for x, y_true in pbar:
            y_pred, h = run_epoch(model,
                                  optimizer,
                                  loss,
                                  x, device, y_true, h)
            predict_true.append(y_true)
            predict_predict.append(y_pred)
            # if log:
            #     pbar.set_description(f"{epoch + 1}/{epochs_count} {score_func(th.cat(predict_true), th.cat(predict_predict))}")
    
        predict_true = th.cat(predict_true)
        predict_predict = th.cat(predict_predict)
        score_train = score_func(predict_predict, predict_true)
        del predict_predict, predict_true
        predict_true = []
        predict_predict = []

        for x, y_true in loader_val:
            optimizer.zero_grad()
            x = x.to(device).detach()
            if h is None:
                y_pred = model.forward(x)
            else:
                y_pred, h = model.forward(x, h)
            predict_true.append(y_true)
            predict_predict.append(y_pred.detach())

        predict_true = th.cat(predict_true)
        predict_predict = th.cat(predict_predict)
        score_val = score_func(predict_predict, predict_true)
        if log:
            print(f"\r Train:{score_train} Val:{score_val}")
        history["train"].append(score_train)
        history["val"].append(score_train)

    return history, model
