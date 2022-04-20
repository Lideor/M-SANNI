# Normalize,aug,search
# -*- coding: utf-8 -*-

import zipfile

import numpy as np
import pandas as pd
import random
from pathlib import Path
import multiprocess
from tqdm import tqdm

from matrixprofile.algorithms import mpdist_vector
from matrixprofile.algorithms.snippets import snippets
from sklearn import preprocessing
import json
import os
import tensorflow as tf
from Preprocess.const import *

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# from API import Image


def augmentation(data: pd.DataFrame, e=0.01):
    """
    Увеличение и балансировка соседей.
    Все соседи сниппетов увеличиваются до количеста соседей у сниппета с максиальным fraction
    :param data: dataframe, в котором хранятся сниппеты и их соседи
    :param e: 0<e<1 процент, на который можно сдвинуть точку
    :return: возвращается dataframe той же структуры, но со сбалансированными соседями
    """
    subseq_count = [(i, len(np.array(data.neighbors.iloc[i]))) for i in range(0, len(data.neighbors))]
    max_subseq_count = max([subseq_count[i][1] for i in range(0, len(subseq_count))])

    new_neighbors_all = []
    for cl in range(0, len(data.neighbors)):
        if subseq_count[cl][1] == max_subseq_count:
            new_neighbors_all.append(data.neighbors[cl].copy())
            continue
        neighbors = data.neighbors[cl].copy()
        need_new_neighbors = (max_subseq_count - subseq_count[cl][1])
        need_double_new = need_new_neighbors - subseq_count[cl][1] if need_new_neighbors - subseq_count[cl][
            1] > 0 else 0
        need_new_neighbors -= need_double_new
        for i in range(0, need_new_neighbors):
            new_neighbor = neighbors[i]
            new_neighbor[random.randint(0, len(neighbors[i]) - 1)] *= 1 + random.uniform(-e, e)
            neighbors.append(new_neighbor)
            if need_double_new > 0:
                new_neighbor = neighbors[i]
                new_neighbor[random.randint(0, len(neighbors[i]) - 1)] *= 1 + random.uniform(-e, e)
                neighbors.append(new_neighbor)
                need_double_new -= 1
        new_neighbors_all.append(neighbors)

    data['neighbors'] = new_neighbors_all
    return data


def create_dataset(size_subsequent: int, dataset: Path, snippet_count=0):
    """
    Создает zip архив в директории датасета с размеченными датасетами
    :param size_subsequent: Размер подпоследовательности
    :param dataset: Директория датасета
    :param snippet_count: минимальный fraction
    :return Возращает колличество сниппетов
    """
    """if not os.path.isdir("{0}/dataset".format(dataset)):
        os.mkdir("{0}/dataset".format(dataset))
    p = Path(dataset + "/data_origin.txt")
    data = np.loadtxt(p)
    data_norm = normalize(data)
    dataset = "{0}/dataset".format(dataset)
    X = []
    y = []

    for i in range(0, len(data) - size_subsequent - 1):
        X.append(json.dumps(data_norm[i:i + size_subsequent - 1].tolist()))
        y.append(json.dumps(data_norm[i + size_subsequent - 1]))

    y = np.array(y)

    print("создал архив")

    filename = 'clear'
    pd.DataFrame({"X": X, "y": y}).to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')
    del X
    """

    p = Path(dataset / DATA_FILE_NAME)
    data = np.loadtxt(p)
    print(len(data.shape))
    if len(data.shape) < 2:
        data = data.reshape(-1, 1)
    data_norm = normalize(data)
    np.savetxt(dataset / NORM_DATA_FILE_NAME, data_norm)
    print("Начал поиск сниппетов", __name__)
    max_snippet = -1
    for idx, data in enumerate(data_norm.T):
        if snippet_count == 0:
            distant = get_distances(ts=data, snippet_size=size_subsequent)
            count_snippet = get_count(distances=distant,
                                      snippet_size=size_subsequent,
                                      len_ts=len(data))
        else:
            count_snippet = snippet_count
        if count_snippet > max_snippet:
            max_snippet = count_snippet
        print(f"Для {idx + 1} признака найденно снипеттов:{count_snippet}")
        snippet_list = search_snippet(data=data,
                                      snippet_count=snippet_count,
                                      size_subsequent=size_subsequent)
        snippet_list.snippet = snippet_list.snippet.apply(lambda x: json.dumps(x.tolist()))
        snippet_list.to_csv(dataset / SNIPPET_FILE_NAME.format(idx + 1), compression='gzip')

    """
    X_classifier = []
    y_classifier = []
    for i, item in snippet_list.iterrows():
        for neighbour in item.neighbors:
            if len(neighbour) == size_subsequent:
                X_classifier.append(json.dumps(np.array(neighbour[:-1]).tolist()))
                y_classifier.append(item["key"])
    print("Создал датасет классификатора")

    y_classifier = tf.keras.utils.to_categorical(np.array(y_classifier))
    filename = 'classifier'
    pd.DataFrame({"X": X_classifier, "y": y_classifier.tolist()}) \
        .to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')
    del X_classifier, y_classifier
    """

    """
    X_predict = []
    y_predict = []

    for i, item in snippet_list.iterrows():
        for neighbour in item.neighbors:
            if len(neighbour) == size_subsequent:
                X_predict.append(json.dumps(np.stack([np.append(neighbour[:-1], [0]), item.snippet]).tolist()))
                y_predict.append(neighbour[-1])

    filename = 'predictor'
    pd.DataFrame({"X": X_predict, "y": y_predict}, columns=["X", "y"]) \
        .to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')

    X_predict = []
    y_predict = []

    for i in range(size_subsequent - 1, len(data) - size_subsequent - 1):
        subsequent = data_norm[i:i + size_subsequent - 1].tolist()
        number = 1
        for j, item in snippet_list.iterrows():
            if i in item.neighbors_index:
                number = i
                break
        X_predict.append(json.dumps(np.stack([np.array(subsequent),
                                              np.full(size_subsequent - 1, number)]).tolist()))
        y_predict.append(json.dumps(data_norm[i + size_subsequent - 1]))

    print("Создал датасет предсказателя")
    snippet_list.neighbors = snippet_list.neighbors.apply(lambda x: json.dumps(x))
    snippet_list.snippet = snippet_list.snippet.apply(lambda x: json.dumps(x.tolist()))
    snippet_list.to_csv(dataset + "/snippet.csv", )

    del snippet_list
    filename = 'predictor_label'
    pd.DataFrame({"X": X_predict, "y": y_predict}, columns=["X", "y"]) \
        .to_csv(f'{dataset}/{filename}.csv.gz', compression='gzip')
    """

    result = {
        "size_subsequent": size_subsequent,
        "snippet_count": max_snippet
    }

    with open(dataset / CURRENT_PARAMS_FILE_NAME, 'w') as outfile:
        json.dump(result, outfile)
    print("Сохранил сниппеты")
    return max_snippet


def search_snippet(data: np.ndarray, snippet_count: int, size_subsequent: int) -> pd.DataFrame:
    """
    Поиск снипетов
    :param data: Директория временного ряда: str
    :param snippet_count: int
    :param size_subsequent: Размер подпоследовательности - int
    :return: Массив снипеетов - np.ndarray
    """

    snp = snippets(data,
                   num_snippets=snippet_count,
                   snippet_size=size_subsequent)

    arr_snp = []
    for i, item in enumerate(snp):
        dict_ = {"key": i,
                 "snippet": item['snippet'],
                 "fraction": item['fraction']}
        neighbors = []
        index = []
        for neighbor in item['neighbors']:
            neighbors.append(data[neighbor:neighbor + size_subsequent].tolist())
            index.append(neighbor)
        dict_["neighbors"] = neighbors
        dict_["neighbors_index"] = index
        arr_snp.append(dict_)
        del item

    df = pd.DataFrame(arr_snp, columns=arr_snp[0].keys())
    df = augmentation(df)
    return df


def normalize(sequent: np.ndarray) -> np.ndarray:
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(sequent)
    return x_scaled


def get_distances(ts, snippet_size):
    window_size = np.floor(snippet_size / 2)
    indices = np.arange(0, len(ts) - snippet_size, snippet_size)
    distances = []

    f = lambda i: mpdist_vector(ts=ts, ts_b=ts[i:(i + snippet_size - 1)], w=int(window_size))
    pool_obj = multiprocess.Pool(4)
    distances = pool_obj.map(f, indices)
    pool_obj.close()
    distances = np.array(distances)
    del pool_obj

    # for i in tqdm(indices):
    #     distance = mpdist_vector(ts, ts[i:(i + snippet_size - 1)], int(window_size))
    #     distances.append(distance)

    return np.array(distances)


def get_count(distances, snippet_size, len_ts, max_k=9):
    profilearea = []
    indices = np.arange(0, len_ts - snippet_size, snippet_size)
    minis = np.inf
    for n in range(max_k):
        minims = np.inf
        for i in range(len(indices)):
            s = np.sum(np.minimum(distances[i, :], minis))

            if minims > s:
                minims = s
                index = i

        minis = np.minimum(distances[index, :], minis)
        profilearea.append(np.sum(minis))
    change = -np.diff(profilearea)

    for i in range(2, len(change)):
        count = (np.trapz(change[:i], dx=1) - np.trapz(change[:i - 1])) / (np.trapz(change[:i], dx=1) + 1)
        if count < 0.15:
            return i
    return len(change)


def smape(a, f):
    return 100/len(a) * np.sum(np.abs(f-a) / ((np.abs(a) + np.abs(f))/2))


def get_score(y_true,y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    loss={}
    loss["mse"] = mean_squared_error(y_true, y_predict)
    loss["rmse"] = mean_squared_error(y_true, y_predict, squared=False)
    loss["mae"] = mean_absolute_error(y_true, y_predict)
    loss["mape"] = mean_absolute_percentage_error(y_true+1, y_predict+1)
    loss["smape"] = smape(y_true+1, y_predict+1)
    loss["r2"] = r2_score(y_true+1, y_predict+1)
    return loss
