# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:36:03 2017

@author: bakan
"""

"""
neuralnetwork：入力層、中間層、出力層の3層のモデル
neuralnetwork_kai：入力層、中間層、中間層、出力層の4層のモデル
neuralnetwork_final:中間層の層数を変更できるモデル
使いたいモデルのコメント文を外してください
"""
from neuralnetwork import *
#from neuralnetwork_kai import *
#from neuralnetwork_final import *
import time
import numpy as np

if __name__ == '__main__':
    #学習データ
    X = np.loadtxt('second.csv', delimiter=',', encoding='utf-8')

    print(X)
    #X = numpy.array([[0, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1]])
    #教師データ
    T = np.loadtxt('tsecond.csv', delimiter=',', encoding='utf-8_sig')
    
    print(T)
    #T = numpy.array([[0], [0], [1], [0], [1], [1]])
    #データ数
    N = X.shape[0]
    #入力データ
    inputdata = np.loadtxt('first.csv', delimiter=',', encoding='utf-8')
    #入力ノード数
    input_size = inputdata.shape[1]
    #中間層ノード数
    hidden_size = 4
    #中間層の層数
    noml = 3
    #出力ノード数
    output_size = 4
    #学習率
    epsilon = 0.1
    #エポック数
    epoch = 1000
    
    starttime = time.time()
    """
    finalを使う場合はnomlが引数に入ってるものを利用、それ以外はnomlが引数に入っていないものを使用
    """
    nn = Neural(input_size, hidden_size, output_size)
    #nn = Neural(input_size, hidden_size, output_size,noml)
    nn.train(X, T, epsilon, epoch)
    #nn.train(X, T, epsilon, epoch,noml)

    Y = nn.predict(X)
    #Y = nn.predict(inputdata,noml)
    
    endtime = time.time()
    
    print(Y)
    np.savetxt('nnsecondfirst.csv', Y, delimiter=',', fmt='%f')
    print(endtime-starttime)
    
    """
    大まかな関数の説明はneuralnetwork.pyに書いておきます
    """
    
