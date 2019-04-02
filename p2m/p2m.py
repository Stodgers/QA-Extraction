#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 10:06
# @Author  : Stodgers
# @Site    : 
# @File    : p2m.py
# @Software: PyCharm

#读入原始数据

csv_path = 'testt.csv'
import jieba
import json
from jieba import analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import re
import sys
import os
path1=os.path.abspath('.')
path2=os.path.abspath('../API/SentenceSimilarity-master')
path3=os.path.abspath('../API/Cluster')
#print(path2)
sys.path.append(path2)
sys.path.append(path3)
from sim_cilin import *
from sim_hownet import *
from sim_simhash import *
from sim_tokenvector import *
from sim_vsm import *
from cluster import *
# In[4]:

dic_word_path = 'all_filter.txt'
filter_word = []
def dic_add(dic_word_path):
    filter_word_t = open(dic_word_path, encoding='utf8')
    for i in filter_word_t.readlines():
        t = i.strip('\n')
        # print(t)
        filter_word.append(t)
        jieba.add_word(t)
dic_add(dic_word_path)

def data_loader(csv_path):
    ts = open(csv_path, errors='ignore')
    ts = pd.read_csv(ts).values.tolist()
    return ts

temp_data_loader = data_loader(csv_path)
print("temp_data_loader: ",len(temp_data_loader))

def text_filter(temp):
    tec = []
    for i in temp:
        text_i = str(i[0]).lower()
        text_i = text_i.replace(' ', '')
        text_i = text_i.replace('　', '')
        ticc = text_i
        if len(text_i) <= 2 or len(text_i) >= 50: continue
        #         segs = jieba.cut(text_i)
        #         segs = list(filter(lambda x:not x.isalpha(),segs))
        #         segs = list(filter(lambda x:not x.isdigit(),segs))
        #         segs = list(filter(lambda x:x not in bd,segs))
        tt = ''.join(re.findall(r'[\u4e00-\u9fa5]', str(i[0])))
        if len(tt) < 3: continue
        segs = jieba.cut(tt)
        segs = list(segs)
        flag = 0
        for j in segs:
            if j in filter_word:
                flag = 1
                break
        if flag: continue
        tec.append((ticc, i[1]))
    return tec

text_filter_temp = text_filter(temp_data_loader)  # 86223 len(temp)
print("text_filter_temp",len(text_filter_temp))

def text_calc_merge(temp):
    tec = []
    dic = {}
    dic_calc = {}
    for i in temp:
        Q = i[0]
        Q = Q.replace(' ', '')
        Q = Q.replace('　', '')
        A = i[1]
        if Q not in dic:
            dic[Q] = A
        if Q not in dic_calc:
            dic_calc[Q] = 1
        else:
            dic_calc[Q] += 1
    tec = [(k, v) for k, v in dic.items()]
    tec = sorted(tec, key=lambda x: x[0])
    tec_calc = [(k, v) for k, v in dic_calc.items()]
    return tec, tec_calc

text_calc_merge_temp, temp_calc = text_calc_merge(text_filter_temp)
print("text_calc_merge_temp",len(text_calc_merge_temp))

def word_process_sim(temp):
    cilin = SimCilin()
    hownet = SimHownet()
    simhash = SimHaming()
    simtoken = SimTokenVec()
    simvsm = SimVsm()
    len_temp = len(temp)
    ans_temp = []
    k = 0
    for i in range(len_temp - 1):
        flag = 0
        tex = temp[i][0]
        if i + 10 < len_temp:
            endd = i + 10
        else:
            endd = len_temp
        for j in range(i + 1, endd):
            texj = temp[j][0]
            try:
                if cilin.distance(tex, texj) >= 0.9:
                    flag = 1
                    break
                if simtoken.distance(tex, texj) >= 0.9:
                    flag = 1
                    break
            except Exception:
                pass

        if flag == 0:
            ans_temp.append((temp[i][0], temp[i][1]))
            if i % 5 == 0:
                t_jd = 100 * i / (len_temp - 1)
                print("{:.2f}%".format(100 * i / (len_temp - 1)) + " " + str(k) + " " + str(i), end='\r')
                print("{:.2f}%".format(100 * i / (len_temp - 1)) + " " + str(k) + " " + str(i), end='\r')
            k += 1
    return ans_temp

ts_ans = []
ts_ans = word_process_sim(text_calc_merge_temp)
print("ts_ans",len(ts_ans), '\n')

cl = cluster(10, ts_ans)
cl.disp()



