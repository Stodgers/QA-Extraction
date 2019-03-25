#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 10:06
# @Author  : Stodgers
# @Site    : 
# @File    : p2m.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import jieba
import re
import sys
sys.path.append(r'..\SentenceSimilarity-master')
from sim_cilin import *
from sim_hownet import *
from sim_simhash import *
from sim_tokenvector import *
from sim_vsm import *


#读入原始数据
csv_path = ''
def text_loader(csv_path):
    try:
        f = open(csv_path,errors='ignore')
    except Exception:
        print('Load file error...\n'
              '1.please use CSV as file\'s suffix\n'
              '2.Please adjust the data format according to the instructions\n'
              )
    df = pd.read_csv(f,encoding='utf-8')
    temp = np.asarray(df)
    len_temp = len(temp)
    temp_ans = []
    for i in range(len_temp):
        temp_ans.append((temp[i][0],temp[i][1]))
    temp_ans = sorted(temp_ans,key=lambda x:str(x[0]),reverse=True)
    return temp_ans
temp = text_loader(csv_path)
print(len(temp))

#把完全过滤词汇添加到词典中
all_filter_word_path = ''
def dic_add(dic_word,all_filter_word_path):
    all_filter_word = []
    filter_word_t = open(all_filter_word_path,encoding='utf-8',errors='ignore')
    for i in filter_word_t.readlines():
        t = i.strip('\n')
        #print(t)
        jieba.add_word(t)
        all_filter_word.append(t)
    return all_filter_word
all_filter_word = dic_add(all_filter_word_path)

#全过滤QA
def text_filter(temp):
    tec = []
    for i in temp:
        text_i = str(i[0]).lower()
        text_i = text_i.replace(' ','')
        text_i = text_i.replace('　', '')
        ticc = text_i
        if len(text_i)<=2 or len(text_i)>=50:continue
        tt = ''.join(re.findall(r'[\u4e00-\u9fa5]',str(i[0])))
        if len(tt)<4:continue
        segs = jieba.cut(tt)
        segs = list(segs)
        flag = 0
        for j in segs:
            if j in all_filter_word:
                flag=1
                break
        if flag:continue
        tec.append((ticc,i[1]))
    return tec
temp = text_filter(temp)
print(len(temp))

#问频统计、同问合并
def text_calc_filter(temp):
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
        else:dic_calc[Q] += 1
    tec = [(k,v) for k,v in dic.items()]
    tec_calc = [(k,v) for k,v in dic_calc.items()]
    return tec,tec_calc
temp,temp_calc = text_calc_filter(temp)

#读取部分过滤词典
#假如某句话包含无意义套话，去除之后若过短不能构成完整的主谓宾结构则删除该条
select_filter_word_path = ''
def load_select_filter_word(select_filter_word_path):
    ans = []
    filter_word_t = open(select_filter_word_path, encoding='utf-8', errors='ignore')
    for i in filter_word_t.readlines():
        t = i.strip('\n')
        #print(t)
        ans.append(t)
    return ans
bd = load_select_filter_word(select_filter_word_path)

