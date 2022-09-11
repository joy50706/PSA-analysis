# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:19:12 2022

@author: lenovo
"""


import pandas as pd
import matplotlib.pylab as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau,rankdata
from scipy.stats import *
from pylab import rcParams
import os
from sklearn.cluster import AgglomerativeClustering
from datetime import date
from sklearn.metrics import roc_curve, auc,roc_auc_score,precision_recall_curve,average_precision_score

colors = [ '#989494','#F56639','#F6C438','#9BEE40', '#41BCED','#41F1ED','#EF3FF3','#EA44A3']
font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 5,
}
lw = 1
method_dict={'ada':'AdaBoost','gbr':'GBDT','lr':'LR','rf':'RF','sgd':'SGD','svm':'SVM','xgb':'XGBoost'}
fs_dict={'no':'no FS','mrmr':'mRMR','chi2':'Chi-square','f':'F score','mi':'MI','rfe-rf':'RFE','sfs-rf':'SFS'}

def draw_line(data,labels,line_names,save_name):
    fig = plt.figure(figsize=(12,4))
    plt.style.use('ggplot')
    ax = fig.add_subplot(1,1,1) 
    for i in range(len(line_names)):
        y=data[line_names[i]].values
        plt.plot(labels, y, 'o-', color=colors[i], label=line_names[i],ms=8)
    # plt.xticks(list(range(1,len(labels)+1)),labels)
    plt.ylim([0.55, 0.95])
    plt.tight_layout()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.legend(loc="upper left")
    plt.savefig(save_name,dpi=600)
    plt.show()


def draw_roc_pr(data,line_names,save_name):
    fig = plt.figure(figsize=(6/2.54,5/2.54))
    plt.style.use('default')
    ax = fig.add_subplot(1,1,1) 

    print(line_names)
    for i in range(len(line_names)):
        temp=data[i]
        label=temp[2]
        pred=temp[3]
        # print(label)
        # print(pred)
        fpr, tpr, thresh = roc_curve(label, pred)
        auc_value = temp[-1]
        plt.plot(fpr,tpr,label=f"{fs_dict[line_names[i]]}+{method_dict[temp[1]]} (AUC={auc_value:.2f})",lw=lw, color=colors[i])
    # plt.xticks(list(range(1,len(labels)+1)),labels)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.tight_layout()
    # plt.grid()
    plt.xticks([0,0.5,1],fontsize=6, family='Arial')
    plt.yticks([0,0.5,1],fontsize=6, family='Arial')
    plt.xlabel('1-Specitivity',fontsize=7, family='Arial')
    plt.ylabel('Sensitivity',fontsize=7, family='Arial')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.legend(prop=font1,loc="lower right")
    plt.savefig(save_name+'_roc_new.pdf',dpi=600)
    plt.show()
    
    fig = plt.figure(figsize=(6/2.54,5/2.54))
    plt.style.use('default')
    ax = fig.add_subplot(1,1,1) 

    print(line_names)
    for i in range(len(line_names)):
        temp=data[i]
        label=temp[2]
        pred=temp[3]
        # print(label)
        # print(pred)
        prec, recall, _ = precision_recall_curve(label, pred)
        auc_value = average_precision_score(label, pred)
        # plt.plot(recall, prec, color=colors[i])
        plt.plot(recall, prec,label=f"{fs_dict[line_names[i]]}+{method_dict[temp[1]]} (AUPR={auc_value:.2f})",lw=lw, color=colors[i])
    # plt.xticks(list(range(1,len(labels)+1)),labels)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    # plt.grid()
    plt.tight_layout()
    plt.xticks([0,0.5,1],fontsize=6, family='Arial')
    plt.yticks([0,0.5,1],fontsize=6, family='Arial')
    plt.xlabel('1-Specitivity',fontsize=7, family='Arial')
    plt.ylabel('Sensitivity',fontsize=7, family='Arial')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.legend(prop=font1,loc="lower right")
    plt.savefig(save_name+'_pr_new.pdf',dpi=600)
    plt.show()

feature_name=['chi2','f','mi','mrmr','rfe-rf','sfs-rf']
methods=['ada','gbr','lr','rf','sgd','svm','xgb']
all_score=[]

all_score=[]
temp=['','','','',-1]
df=pd.read_csv(f'new_results/risk_prediction/all_features_new_data.csv')
temp_score=[]
for each_method in methods:  
    y_pred=df[each_method].values
    y_true=df['true'].values
    temp_score=roc_auc_score(y_true, y_pred)
    if temp_score>temp[-1]:
        temp=[0,each_method,y_true,y_pred,temp_score]
    
all_score.append(temp)
for each_name in feature_name:
    
    temp=['','','','',-1]
    if feature_name=='mrmr':
        for n_features in range(5,105,5):
            df=pd.read_csv(f'new_results/risk_prediction/new_data_{each_name}_features_{n_features}.csv')
            temp_score=[]
            for each_method in methods:  
                y_pred=df[each_method].values
                y_true=df['true'].values
                temp_score=roc_auc_score(y_true, y_pred)
                if temp_score>temp[-1]:
                    temp=[n_features,each_method,y_true,y_pred,temp_score]
    else:
        for n_features in range(5,105,5):
            df=pd.read_csv(f'new_results/risk_prediction/new_data_{each_name}_features_{n_features}.csv')
            temp_score=[]
            for each_method in methods:  
                y_pred=df[each_method].values
                y_true=df['true'].values
                temp_score=roc_auc_score(y_true, y_pred)
                if temp_score>temp[-1]:
                    temp=[n_features,each_method,y_true,y_pred,temp_score]
    all_score.append(temp)

    
# score_df=pd.DataFrame(all_score,columns=methods)
# labels=list(range(5,105,5))
draw_roc_pr(all_score,['no']+feature_name,f'final_fig/new_risk_predict_best')
# df.to_csv('new_results/mrmr_cv_label_stack.csv',index=None)