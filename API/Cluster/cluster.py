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

def jieba_tockenize(text):
    return jieba.lcut(text)

tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tockenize,sublinear_tf=True,lowercase=False)

'''
tokenizer: extract func
lowercase：lower
'''


class cluster:
    def __init__(self,path,num):
        self.path = path
        self.num_clusters = num
        self.text_list,self.tt = self.data_load()
        self.tfidf_matrix = self.tf_idf()
        self.result = self.Kmeans()

    def data_load(self):
        data = open(self.path,errors='ignore')
        data = pd.read_csv(data)
        tes = [i for i in data]
        text_list = [i for i in data.iloc[:,0]]
        return text_list,data

    def tf_idf(self):
        tfidf_vectorizer.fit(self.text_list)
        vbchart = dict(map(lambda t:(t[1],t[0]),tfidf_vectorizer.vocabulary_.items()))
        tfidf_matrix = tfidf_vectorizer.transform(self.text_list)
        return tfidf_matrix

    def Kmeans(self):
        km_cluster = KMeans(n_clusters=self.num_clusters,max_iter=200,
                            n_init=40,tol=1e-6,init='random',n_jobs=-1
                            )
        result = km_cluster.fit_predict(self.tfidf_matrix)
        return result

    def disp(self):
        keyword_list = []
        res = self.result
        tt = self.tt
        for i in range(self.num_clusters):
            text = ''
            for j in range(len(res)):
                if res[j] == i:
                    text += self.text_list[j]
            keywords = analyse.textrank(text)
            keyword_list.append(keywords[:3])
        ans = [(res[i],keyword_list[res[i]],tt.iloc[i,0],tt.iloc[i,1]) for i in range(len(res))]
        ans = sorted(ans,key=lambda x:x[0])
        dff = pd.dataFrame(ans)
        dff.to_csv('cluster_ans.csv',index=False,header=False)
