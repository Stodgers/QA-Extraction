#!/usr/bin/env python3
# coding: utf-8
# File: func_use.py

from sim_cilin import *
from sim_hownet import *
from sim_simhash import *
from sim_tokenvector import *
from sim_vsm import *

def testt():
    cilin = SimCilin()
    hownet = SimHownet()
    simhash = SimHaming()
    simtoken = SimTokenVec()
    simvsm = SimVsm()

    while 1:
        text1 = input('enter sent1:').strip()
        text2 = input('enter sent2:').strip()
        print('cilin', cilin.distance(text1, text2))
        print('hownet', hownet.distance(text1, text2))
        print('simhash', simhash.distance(text1, text2))
        print('simtoken', simtoken.distance(text1, text2))
        print('simvsm', simvsm.distance(text1, text2))
#testt()
def word_process_sim(temp):
    cilin = SimCilin()
    hownet = SimHownet()
    simhash = SimHaming()
    simtoken = SimTokenVec()
    simvsm = SimVsm()
    len_temp = len(temp)
    ans_temp = []
    k = 0
    for i in range(len_temp-1):
        flag = 0
        tex = temp[i][0]
        if i+10<len_temp:
            endd = i+10
        else: endd = len_temp
        for j in range(i+1,endd):
            texj = temp[j][0]
            try:
                if cilin.distance(tex,texj)>=0.8:
                    flag = 1
                    break
                if cilin.distance(tex,texj)>=0.8:
                    flag = 1
                    break
            except Exception:pass
        if flag==0:
            ans_temp.append((temp[i][0],temp[i][1]))
            if i%100==0:
                t_jd = 100*i/(len_temp-1)
                print("{:.2f}".format(t_jd)+" "+str(k)+" "+str(i),end='\r')
            k+=1

    return ans_temp