# -*- coding:utf-8 -*-

# @Author zpf

from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from decimal import *
import csv
import numpy as np
import xgboost as xgb
import pandas


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

    Y = train[:, -1]

    test_avg_predict = np.zeros(len(test_normalized))
    test_predict_num = 0

    # 交叉验证
    kf = KFold(n_splits=20)
    for train_index, val_index in kf.split(train_normalized):
        x_train, x_val = train_normalized[train_index], train_normalized[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        clr = linear_model.LinearRegression()
        gbr = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1,
                                        max_depth=3)
        clr.fit(x_train, y_train)
        gbr.fit(x_train, y_train)

        y_predict = clr.predict(x_val)
        y_predict_gbr = gbr.predict(x_val)

        # print(y_predict)
        # print(y_val)
        val_loss = sum((y_predict - y_val) ** 2) / (2 * len(y_predict))
        val_loss_gbr = sum((y_predict_gbr - y_val) ** 2) / (2 * len(y_predict_gbr))
        print(str(val_loss) + '\t' + str(val_loss_gbr))
        print(train_index, val_index)
        if val_loss < 0.7:
            test_predict = clr.predict(test_normalized)
            test_avg_predict += test_predict
            test_predict_num += 1

    test_avg_predict = test_avg_predict/test_predict_num
    test_result_list = test_avg_predict.tolist()
    # print(test_result_list)
    final_result = []
    getcontext().prec = 4
    for e in test_result_list:
        temp_result = Decimal(e) * Decimal(1)
        final_result.append(temp_result)
    # print(final_result)

    csv_file = open('./result.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    # for i in range(len(final_result)):
    final_result = map(lambda x: [x], final_result)
    writer.writerows(final_result)
    csv_file.close()


if __name__ == "__main__":
    train_set, test_set = data_process()
    predict(train_set, test_set)
