# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 08:54:23 2022

@author: lenovo
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
feature_name=['chi2','mi','f','mrmr','rfe']
methods=['ada','gbr','lr','rf','sgd','svm','xgb']
new_feature_name=['CHISquare','MI','F','mRMR','RFE']

feature_dict={}
for i in range(len(feature_name)):
    feature_dict[feature_name[i]]=i

method_dict={}
for i in range(len(methods)):
    method_dict[methods[i]]=i

data_matrix=np.zeros([len(feature_name),len(methods)])
file_list=os.listdir('new_results_downsampling_data/feature_importance')
for each_file in file_list:
    temp=each_file.split('_')
    i=feature_dict[temp[1]]
    j=method_dict[temp[2]]
    data_matrix[i,j]+=1

data_matrix=data_matrix.tolist()
new_data=[]
for i in range(len(data_matrix)):
    new_data.append([new_feature_name[i]]+data_matrix[i])


new_data_for_save_df=pd.DataFrame(new_data,columns=['FS']+\
                                            ['AdaBoost','GBDT','LR','RF','SGD','SVM','XGBoost'])
new_data_for_save_df.to_csv('new_results_downsampling_data/features_methods_for_heatmap.csv',index=None)