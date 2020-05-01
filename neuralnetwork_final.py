# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:35:12 2020

@author: bakan
"""


import numpy
import math

class Neural:

    # コンストラクタ
    def __init__(self, n_input, n_hidden, n_output,noml):
        
        #for文で必要な数の中間層を配列で格納
        self.hidden_weight = numpy.random.random_sample((n_hidden, n_input + 1))
        if noml > 2:
            self.middle_weight = []
            for i in range(noml - 1):
                self.middle_weight.append(numpy.random.random_sample((n_hidden, n_hidden + 1)))
            self.output_weight = numpy.random.random_sample((n_output, n_hidden + 1))
        elif noml > 1:
            self.middle_weight = (numpy.random.random_sample((n_hidden, n_hidden + 1)))
            self.output_weight = numpy.random.random_sample((n_output, n_hidden + 1))
        else:
            self.output_weight = numpy.random.random_sample((n_output, n_hidden + 1))
            


# public method
    def train(self, X, T, epsilon, epoch,noml):
        self.error = numpy.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            for i in range(N):
                x = X[i, :]
                t = T[i, :]

                self.__update_weight(x, t, epsilon,noml)

            self.error[epo] = self.__calc_error(X, T,noml)


    def predict(self, X,noml):
        N = X.shape[0]
        Y = [0]*N
        for i in range(N):
            x = X[i, :]
            z, y, b = self.__forward(x,noml)

            Y[i] = y
            
        return (Y)

    def __sigmoid(self, arr):
        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)

    #中間層が1、2、3以上の時で場合分けしている
    def __forward(self, x,noml):
        
        # z: output in hidden layer, y: output in output layer
        z = self.__sigmoid(self.hidden_weight.dot(numpy.r_[numpy.array([1]), x]))
        if noml > 2:
            b=[0]*(noml - 1)
            b[0] = self.__sigmoid(self.middle_weight[0].dot(numpy.r_[numpy.array([1]), z]))
            for i in range(noml - 2):
                b[i + 1] = self.__sigmoid(self.middle_weight[i + 1].dot(numpy.r_[numpy.array([1]), b[i]]))
                
            y = self.__sigmoid(self.output_weight.dot(numpy.r_[numpy.array([1]), b[i + 1]]))
            
        elif noml > 1:
            b = []
            b = self.__sigmoid(self.middle_weight.dot(numpy.r_[numpy.array([1]), z]))
            y = self.__sigmoid(self.output_weight.dot(numpy.r_[numpy.array([1]), b]))
    
        else:   
            b = []
            y = self.__sigmoid(self.output_weight.dot(numpy.r_[numpy.array([1]), z]))
            
        return (z, y, b)

    #配列計算簡略化のため、中間層とシグモイド計算した配列を逆転して計算、これも中間層数1,2,3以上で場合分けしている
    def __update_weight(self, x, t, epsilon,noml):
        z, y, b = self.__forward(x,noml)
        
        if noml > 2:
            self.middle_weight.reverse()
            b.reverse()
            middle_delta = [0]*(noml - 1)
            output_delta = (y - t) * y * (1.0 - y)
            self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), b[0]]
            
            middle_delta[0] = (self.output_weight[:, 1:].T.dot(output_delta)) * b[0] * (1.0 - b[0])
            self.middle_weight[0] -= epsilon * middle_delta[0].reshape((-1, 1)) * numpy.r_[numpy.array([1]), b[1]]
            
            for i in range(noml - 2):
                d = noml - 2
                d -= 1    
                fake_middle_delta = middle_delta[i + 1]
                fake_middle_weight = self.middle_weight[i]
                fake_middle_delta = (fake_middle_weight[:, 1:].T.dot(middle_delta[i])) * b[i + 1] * (1.0 - b[i + 1])
                if  d > 0:
                    self.middle_weight[i + 1] -= epsilon * fake_middle_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), b[i + 2]]
                else:
                    self.middle_weight[i + 1] -= epsilon * fake_middle_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), z]

            # update hidden_weight
            fake_middle_weight = self.middle_weight[i + 1]
            hidden_delta = (fake_middle_weight[:, 1:].T.dot(fake_middle_delta)) * z * (1.0 - z)
            self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), x]
        
            self.middle_weight.reverse()
            b.reverse()
            
        elif noml > 1:
            
            output_delta = (y - t) * y * (1.0 - y)
            self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), b]

            middle_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * b * (1.0 - b)
            self.middle_weight -= epsilon * middle_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), z]

        # update hidden_weight
            hidden_delta = (self.middle_weight[:, 1:].T.dot(middle_delta)) * z * (1.0 - z)
            self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), x]
            
        
        else:
        # update output_weight
            output_delta = (y - t) * y * (1.0 - y)
            self.output_weight -= epsilon * output_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), z]

        # update hidden_weight
            hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta)) * z * (1.0 - z)
            self.hidden_weight -= epsilon * hidden_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), x]

            print(self.output_weight[:, 1:].T.dot(output_delta))

    def __calc_error(self, X, T,noml):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = T[i, :]

            z, y ,b= self.__forward(x,noml)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0

        return err