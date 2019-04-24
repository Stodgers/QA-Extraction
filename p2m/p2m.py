#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 10:06
# @Author  : Stodgers
# @Site    : 
# @File    : p2m.py
# @Software: PyCharm

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

################################################
################################################
'''数据路径'''
csv_path = '家电问诊q2a.csv'

'''全过滤词词典'''
dic_word_path = 'all_filter.txt'

'''聚类簇,数目'''
cluster_num = 20

'''相似问合并参数,0.9*0.9=0.81'''
sim_seed = 0.45
################################################
################################################


'''把全过滤词硬添加进分词词典,过滤匹配'''
filter_word = []
def dic_add(dic_word_path):
    filter_word_t = open(dic_word_path, encoding='utf8')
    for i in filter_word_t.readlines():
        t = i.strip('\n')
        # print(t)
        filter_word.append(t)
        jieba.add_word(t)
dic_add(dic_word_path)

'''数据加载,容错'''
def data_loader(csv_path):
    with open(csv_path, errors='ignore') as ts:
        let = ts.read(10)
        tt = len(''.join(re.findall(r'[\u4e00-\u9fa5]', let)))
        if tt != 0:
            ts = open(csv_path, errors='ignore')
            df_list = pd.read_csv(ts).values.tolist()
    with open(csv_path, errors='ignore', encoding='utf-8') as ts:
        let = ts.read(10)
        tt = len(''.join(re.findall(r'[\u4e00-\u9fa5]', let)))
        if tt != 0:
            ts = open(csv_path, errors='ignore', encoding='utf-8')
            df_list = pd.read_csv(ts).values.tolist()
    with open(csv_path, errors='ignore', encoding='utf-8-sig') as ts:
        let = ts.read(10)
        tt = len(''.join(re.findall(r'[\u4e00-\u9fa5]', let)))
        if tt != 0:
            ts = open(csv_path, errors='ignore', encoding='utf-8-sig')
            df_list = pd.read_csv(ts).values.tolist()
    return df_list
temp_data_loader = data_loader(csv_path)
print("temp_data_loader: ",len(temp_data_loader))

'''数据过滤'''
def text_filter(temp):
    tec = []
    k = 0
    for i in temp:
        text_i = str(i[0]).lower()
        text_i = text_i.replace(' ', '')
        text_i = text_i.replace('　', '')
        ticc = text_i
        if len(text_i) <= 2 or len(text_i) >= 50: continue
        tt = ''.join(re.findall(r'[\u4e00-\u9fa5]', str(i[0])))#把文字取出来,求长度
        if len(tt) < 3: continue#长度小于3就扔掉,不满足主谓宾的合格句子成分
        segs = jieba.cut(tt)
        segs = list(segs)
        flag = 0
        for j in segs:
            if j in filter_word:
                flag = 1
                break
        if flag: continue
        tec.append([ticc,i[1],k])
        k += 1
    return tec

text_filter_temp = text_filter(temp_data_loader)  # 86223 len(temp)
print("text_filter_temp",len(text_filter_temp)) #输出一下过滤后的长度

'''数据统计(频次),结构拼装()'''
def text_calc_merge(temp):
    dic_index = {}
    dic = {}
    dic_calc = {}
    k = 0
    for i in temp:
        Q = i[0]
        Q = Q.replace(' ', '')
        Q = Q.replace('　', '')
        A = i[1]
        if Q not in dic:
            dic[Q] = A
            dic_index[k] = i[2]
            k += 1
        if Q not in dic_calc:
            dic_calc[Q] = 1
        else:
            dic_calc[Q] += 1

    tec = [[k, v] for k, v in dic.items()]
    tec_calc = [[k, v] for k, v in dic_calc.items()]
    for i,v in enumerate(tec):
        tec_calc[i].append(tec[i][1])#把Answer拼接在后面
        tec_calc[i].append(dic_index[i])#把索引拼接在后面
    '''拼接后tec_calc每行的结构->(Q,Num,A,index)'''
    tec = sorted(tec, key=lambda x: x[0], reverse=True)
    tec_calc = sorted(tec_calc, key=lambda x: x[1],reverse=True)#频次排序
    return tec, tec_calc
text_calc_merge_temp, temp_calc = text_calc_merge(text_filter_temp)

print("text_calc_merge_temp",len(text_calc_merge_temp))
print(temp_calc[:2])

'''相似度计算合并相似问'''
def word_process_sim(temp,temp_calc):
    cilin = SimCilin()
    hownet = SimHownet()
    simhash = SimHaming()
    simtoken = SimTokenVec()
    simvsm = SimVsm()
    len_temp = len(temp)
    ans_temp = []
    k = 0

    '''为Q_list添加空存储位置,存储在每行最后一个位置'''
    for i,v in enumerate(temp_calc):
        temp_calc[i].append([])

    for i in range(len_temp - 1):
        temp_q_list = list(set(temp_calc[i][4]))#有重复继承,set化
        flag = 0
        tex = temp_calc[i][0]

        if i + 10 < len_temp:
            endd = i + 10
        else:
            endd = len_temp
        for j in range(i + 1, endd):
            texj = temp_calc[j][0]
            try:
                if cilin.distance(tex, texj)*simtoken.distance(tex, texj) >= sim_seed:
                    temp_calc[j][1] += temp_calc[i][1] #相似问也统计进当前Q之中
                    temp_calc[i][1] = 0                #原句频次置零
                    flag = 1
                    temp_q_list.append(tex)
                    temp_calc[j][4]+=temp_q_list
                    break
            except Exception:
                pass

        if flag == 0:
            ans_temp.append((temp[i][0], temp[i][1]))
            if i % 100 == 0:
                t_jd = 100 * i / (len_temp - 1)
                print("{:.2f}%".format(t_jd) + " " + str(k) + " " + str(i))#, end='\r'#刷新进度
            k += 1
    temp_calc = sorted(temp_calc,key=lambda x:x[0],reverse=True)
    '''结构重新组装'''
    temp_calc = [[i[0],i[2],i[1],i[3],i[4][::-1]] for i in temp_calc if i[1]!=0]
    return temp_calc

ts_ans = []
Qrank= word_process_sim(text_calc_merge_temp,temp_calc)
print("Qrank",len(Qrank), '\n')

cl = cluster(cluster_num, Qrank)
keyword, rank_ans= cl.disp()

rank_temp = []
cu_temp = []
sum = 0
for i,v in enumerate(rank_ans):
    if i==len(rank_ans)-1 or rank_ans[i][0]!=rank_ans[i+1][0] :
        cu_temp.append(rank_ans[i])
        sum += rank_ans[i][4]
        rank_temp.append([sorted(cu_temp,key=lambda x:x[4],reverse=True),sum])
        sum=0
        cu_temp = []
    else:
        cu_temp.append(rank_ans[i])
        sum += rank_ans[i][4]

rank_temp = sorted(rank_temp,key=lambda x:x[1],reverse=True)

ans = []
for i in rank_temp:
    for j in i[0]:
        ans.append(j)
        #print(j)
'''关键词过滤'''
ans_filter_word = []
def ans_dic_add(dic_word_path):
    filter_word_t = open(dic_word_path, encoding='utf8')
    for i in filter_word_t.readlines():
        t = i.strip('\n')
        # print(t)
        filter_word.append(t)
        jieba.add_word(t)
ans_dic_add('dic\\ans_filter.txt')

ans_filtered = []
for k,v in enumerate(ans):
    keyword_segs = ans[k][1].split(',')
    keyword_segs = list(filter(lambda x: x.strip(), keyword_segs))
    keyword_segs = list(filter(lambda x: len(x) > 1, keyword_segs))
    keyword_segs = [i for i in keyword_segs if i not in ans_filter_word]
    q_segs = jieba.lcut(ans[k][2])
    q_segs = list(filter(lambda x: x.strip(),q_segs))
    q_segs = list(filter(lambda x: len(x) > 1,q_segs))
    flag = 0
    for j in q_segs:
        if j in keyword_segs:
            flag = 1
            break
    if flag == 1:
        ans[k][1] = ','.join(keyword_segs)
        ans_filtered.append(v)

df = pd.DataFrame(ans_filtered)
df.to_csv('ans\\Qrank_家电问诊q2a.csv',
          index=False,
          header=['Seed','keyword','Q','A','num','index','Q_list'],
          encoding='utf-8-sig'
          )