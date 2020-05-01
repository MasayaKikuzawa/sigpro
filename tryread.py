# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:14:43 2020

@author: bakan
"""


import csv
import pprint
import numpy

with open('first.csv',encoding="utf-8_sig") as f:
    a = numpy.array(f.read())
    
print(a)