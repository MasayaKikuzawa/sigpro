# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:35:12 2020

@author: bakan
"""


import numpy
import math

class Neural:

    # constructor
    def __init__(self, n_input, n_hidden, n_output):
        self.hidden_weight = numpy.random.random_sample((n_hidden, n_input + 1))
        self.middle_weight = numpy.random.random_sample((n_hidden, n_hidden + 1))
        self.output_weight = numpy.random.random_sample((n_output, n_hidden + 1))


# public method
    def train(self, X, T, epsilon, epoch):
        self.error = numpy.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            for i in range(N):
                x = X[i, :]
                t = T[i, :]

                self.__update_weight(x, t, epsilon)

            self.error[epo] = self.__calc_error(X, T)


    def predict(self, X):
        N = X.shape[0]
        Y = [0]*N
        for i in range(N):
            x = X[i, :]
            z, y, a = self.__forward(x)

            Y[i] = y
            
        return (Y)

# private method
    def __sigmoid(self, arr):
        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)


    def __forward(self, x):
        # z: output in hidden layer, y: output in output layer
        z = self.__sigmoid(self.hidden_weight.dot(numpy.r_[numpy.array([1]), x]))
        a = self.__sigmoid(self.middle_weight.dot(numpy.r_[numpy.array([1]), z]))
        y = self.__sigmoid(self.output_weight.dot(numpy.r_[numpy.array([1]), a]))

        return (z, y, a)

    def __update_weight(self, x, t, epsilon):
        z, y, a = self.__forward(x)

        # update output_weight
        output_delta = (y - t) * y * (1.0 - y)
        self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), a]

        middle_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * a * (1.0 - a)
        self.middle_weight -= epsilon * middle_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), z]

        # update hidden_weight
        hidden_delta = (self.middle_weight[:, 1:].T.dot(middle_delta)) * z * (1.0 - z)
        self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), x]


    def __calc_error(self, X, T):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = T[i, :]

            z, y ,a= self.__forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0

        return err