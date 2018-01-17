# -*- coding:utf-8 -*-

# @Author zpf

from sklearn import linear_model
from sklearn import preprocessing
import csv
import numpy as np


def data_process():
    train_data_dir = "./data/d_train_20180102.csv"
    test_data_dir = "./data/d_test_A_20180102.csv"
    train_data = []
    test_data = []
    with open(train_data_dir, "r", encoding="gb2312") as f1:
        for row in f1:
            temp = row.replace("\n", "")
            temp = temp.split(",")
            for e in temp:
                if e == "":
                    temp[temp.index(e)] = 0.0
                if e == "男":
                    temp[temp.index(e)] = 1.0
                if e == "女":
                    temp[temp.index(e)] = 0.0
                if e == '??':
                    temp[temp.index(e)] = 0.5
            train_data.append(temp)
        train_data = np.array(train_data)
        # print(type(train_data))
        # print(train_data[1, 4:].astype(np.float64))
    train_slice = np.concatenate((train_data[1:, 1:3], train_data[1:, 4:]), axis=1)

    with open(test_data_dir, "r", encoding="gb2312") as f2:
        for row in f2:
            row = row.strip().split(",")
            for e in row:
                if e == "":
                    row[row.index(e)] = 0
                if e == "男":
                    row[row.index(e)] = 1
                if e == "女":
                    row[row.index(e)] = 0
            test_data.append(row)
        test_data = np.array(test_data)
    test_slice = np.concatenate((test_data[1:, 1:3], test_data[1:, 4:]), axis=1)

    return train_slice, test_slice


def predict(train, test):
    train = train.astype(np.float)
    test = test.astype(np.float)
    
    train_normalized = preprocessing.normalize(train[:, :-1], norm='l2')
    test_normalized = preprocessing.normalize(test, norm='l2')

    X = train_normalized[:, :-1]
    Y = train[:, -1]
    train_x = X[:4000, :]
    train_y = Y[:4000]
    val_x = X[4000:, :]
    val_y = Y[4000:]

    clr = linear_model.LinearRegression()
    clr.fit(train_x, train_y)
    y_predict = clr.predict(val_x)
    print(y_predict)
    print(val_y)
    loss = sum((y_predict - val_y) ** 2)/(2 * len(y_predict))
    print(loss)


if __name__ == "__main__":
    train_set, test_set = data_process()
    predict(train_set, test_set)
