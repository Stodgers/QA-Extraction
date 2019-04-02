#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 11:01
# @Author  : Stodgers
# @Site    : 
# @File    : jh.py
# @Software: PyCharm


import sys
import os
import os
path1=os.path.abspath('.')
path2=os.path.abspath('../API/SentenceSimilarity-master')

#print(path2)
sys.path.append(path2)
from sim_cilin import *
from sim_hownet import *
from sim_simhash import *
from sim_tokenvector import *
from sim_vsm import *

cilin = SimCilin()
hownet = SimHownet()
simhash = SimHaming()
simtoken = SimTokenVec()
simvsm = SimVsm()

text1 = '我想喝可乐'
text2 = '有没有可口可乐'
print('cilin', cilin.distance(text1, text2))
print('hownet', hownet.distance(text1, text2))
print('simhash', simhash.distance(text1, text2))
print('simtoken', simtoken.distance(text1, text2))
print('simvsm', simvsm.distance(text1, text2))