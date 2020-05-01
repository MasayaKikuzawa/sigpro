# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:35:12 2017

@author: bakan
"""


import numpy
import math

class Neural:

    # コンストラクタ
    def __init__(self, n_input, n_hidden, n_output):
        self.hidden_weight = numpy.random.random_sample((n_hidden, n_input + 1))
        self.output_weight = numpy.random.random_sample((n_output, n_hidden + 1))

    def train(self, X, T, epsilon, epoch):
        self.error = numpy.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            for i in range(N):
                x = X[i, :]
                t = T[i, :]

                self.__update_weight(x, t, epsilon)

            self.error[epo] = self.__calc_error(X, T)

    #最終的な推定、学習後に入力データを分類
    def predict(self, X):
        N = X.shape[0]
        Y = [0]*N
        for i in range(N):
            x = X[i, :]
            z, y = self.__forward(x)

            Y[i] = y
            
        return (Y)

    # シグモイド関数
    def __sigmoid(self, arr):
        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)

    #入力したデータを順伝搬で出力、誤差計算、学習、推定に利用
    def __forward(self, x):
        # z: output in hidden layer, y: output in output layer
        z = self.__sigmoid(self.hidden_weight.dot(numpy.r_[numpy.array([1]), x]))
        y = self.__sigmoid(self.output_weight.dot(numpy.r_[numpy.array([1]), z]))

        return (z, y)
    
    #重みを計算
    def __update_weight(self, x, t, epsilon):
        z, y = self.__forward(x)

        # update output_weight
        output_delta = (y - t) * y * (1.0 - y)
        self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), z]
        
        # update hidden_weight
        hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * z * (1.0 - z)
        self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), x]

    #誤差計算
    def __calc_error(self, X, T):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = T[i, :]

            z, y = self.__forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0

        return err