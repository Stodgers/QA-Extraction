#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/26 17:02
# @Author  : Stodgers
# @Site    : 
# @File    : cluster.py
# @Software: PyCharm

import jieba
import pandas as pd
import json
from jieba import analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

'''
tokenizer: extract func
lowercase：lower
'''

#data:输入数据,可以传入list、csv和xlsx的文件路径
#num:聚类的数目
#Ps.输入格式为每行为一组QA，例如list['怎么开票？','登陆xx系统开票']
class cluster:
    def __init__(self,num,data):
        self.data = data
        self.num_clusters = num
        self.Q_list,self.tt = self.data_load()
        self.tfidf_matrix = self.tf_idf()
        self.result = self.Kmeans()
    #global data
    def data_load(self):
        if type(self.data) == list:
            pass
        else:
            try:
                suffix = str(self.data).split('.')[1]
                if suffix == 'csv':
                    with open(self.data,errors='ignore') as f:
                        self.data = pd.read_csv(f).values.tolist()
                    #data = open(self.data,errors='ignore').values.tolist()
                elif suffix == 'xlsx':
                     with pd.read_excel(self.data,header = None) as f:
                         self.data = f.values.tolist()
                print('已读取文件: ',suffix)
            except Exception:
                print('文件读取失败！只能读取list、csv、xlsx文件')
                pass
        Q_list = [i[0] for i in self.data]
        return Q_list,self.data

    def tf_idf(self):
        #analyse.set_stop_words('all_filter.txt')
        def jieba_tockenize(text):
            return jieba.lcut(text)
        tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tockenize,
                                           sublinear_tf=True, lowercase=False)
        tfidf_vectorizer.fit(self.Q_list)
        vbchart = dict(map(lambda t:(t[1],t[0]),tfidf_vectorizer.vocabulary_.items()))
        tfidf_matrix = tfidf_vectorizer.transform(self.Q_list)
        return tfidf_matrix

    def Kmeans(self):
        km_cluster = KMeans(n_clusters=self.num_clusters,max_iter=50,
                            n_init=40,tol=1e-6,init='random',n_jobs=-1
                            )
        result = km_cluster.fit_predict(self.tfidf_matrix)
        return result

    def disp(self):
        key_temp = []
        keyword_list = []
        res = self.result
        tt = self.tt
        for i in range(self.num_clusters):
            text = ''
            for j in range(len(res)):
                if res[j] == i:
                    text += self.Q_list[j]
            keywords = analyse.textrank(text)
            #keyword_list.append(",".join(keywords[:3]))
            for i in keywords:
                key_temp.append(i)
            key_temp = list(set(key_temp))
            keyword_list.append(",".join(keywords))
            #keyword_list.append(key_temp[:3])
        #tt[i][0],tt[i][1],tt[i][2],tt[i][3]
        ans = [[res[i],keyword_list[res[i]],tt[i][0],tt[i][1],tt[i][2],tt[i][3],tt[i][4]] for i in range(len(res))]
        ans = sorted(ans,key=lambda x:x[0])
        dff = pd.DataFrame(ans)
        dff.to_csv('ans\\cluster_ans.csv',index=False,header=False,encoding='utf-8-sig')
        return key_temp,ans

'''
csv_path = 'C:\\Users\\Ma\\Desktop\\QA.csv'
cl = cluster(10, csv_path)
cl.disp()
'''

