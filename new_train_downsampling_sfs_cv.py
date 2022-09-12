# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:52:12 2022

@author: lenovo
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:18:54 2022

@author: lenovo
"""



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
# import pymrmr
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2,mutual_info_classif,f_classif
# train_df = pd.DataFrame(X_train, columns = features_name)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import RepeatedKFold
import pymrmr
from sklearn.feature_selection import SequentialFeatureSelector
import os
import numpy as np
import pandas as pd
import random

dir_name='sfs'
for repeats in range(40):
    # train_false_index=list(range(462,len(new_data)))
    # random.shuffle(train_false_index)
    # train_index=list(range(462))+train_false_index[:462]
    # X_train=new_data[train_index]
    # X_test=new_test_data
    # y_test=test_labels
    # y_train=labels[train_index]
    # X_train_for_df=np.hstack((y_train.reshape(len(y_train),1),X_train))
    # train_df = pd.DataFrame(X_train_for_df, columns = ['label']+features_name)
    temp_dir_list=os.listdir('new_results_downsampling_data')
    if str(repeats) not in temp_dir_list:
        os.mkdir('new_results_downsampling_data/'+str(repeats))
    X_train=np.load(f'new_results_downsampling_data/data/X_train_{repeats}.npy')
    X_test=np.load(f'new_results_downsampling_data/data/X_test_{repeats}.npy')
    
    y_train=np.load(f'new_results_downsampling_data/data/y_train_{repeats}.npy')
    y_test=np.load(f'new_results_downsampling_data/data/y_test_{repeats}.npy')  
    train_df=pd.read_csv(f'new_results_downsampling_data/data/X_train_df_{repeats}.csv')

    temp_dir_list=os.listdir('new_results_downsampling_data/'+str(repeats))
    method_list=['ada','gbr','lr','rf','sgd','svm','xgb']
    for each_method in method_list:
        if dir_name+'_'+each_method+'_cv' not in temp_dir_list:
            os.mkdir('new_results_downsampling_data/'+str(repeats)+'/'+dir_name+'_'+each_method+'_cv')
    
    for n_features in range(5,105,5):
    # selected_features_name=pymrmr.mRMR(train_df,'MIQ', n_features)
    # new_index=[]
    # for i in range(len(features_name)):
    #     if features_name[i] in selected_features_name:
    #         new_index.append(i)
    # temp_X_train=X_train[:,new_index]
    # temp_X_test=X_test[:,new_index]

    # selected_features_name=pymrmr.mRMR(train_df,'MIQ', n_features)
    # new_index=[]
    # for i in range(len(features_name)):
    #     if features_name[i] in selected_features_name:
    #         new_index.append(i)
    # temp_X_train=X_train[:,new_index]
    # temp_X_test=X_test[:,new_index]
    # estimator = RandomForestClassifier(max_depth=4, random_state=0)
    # selector = SequentialFeatureSelector(estimator, n_features_to_select=n_features)
    # selector = selector.fit(X_train, y_train)
    # # selected_index=selector.get_support
    # temp_X_train=selector.transform(X_train)
    # temp_X_test=selector.transform(X_test)
    # selected_features_name=[]
    # for each_index in selected_index:
    #     selected_features_name.append(features_name[each_index])
        
        estimator = RandomForestClassifier(max_depth=4, random_state=0)
        selector = SequentialFeatureSelector(estimator, n_features_to_select=n_features)
        selector = selector.fit(X_train, y_train)
    # selected_index=selector.get_support
        temp_X_train=selector.transform(X_train)
        temp_X_test=selector.transform(X_test)
    #     selector=SelectKBest(mutual_info_classif, k=n_features).fit(X_train, y_train)
    # # selected_index=selector.get_support
    #     temp_X_train=selector.transform(X_train)
    #     temp_X_test=selector.transform(X_test)        
        rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    ####rf
        count=0
        for train_index, test_index in rkf.split(temp_X_train):
            new_X_train=temp_X_train[train_index]
            new_X_test=temp_X_train[test_index]
            new_y_train=y_train[train_index]
            new_y_test=y_train[test_index]
            model_rf=RandomForestClassifier(max_depth=4, random_state=0).fit(new_X_train, new_y_train)
            y_pred_rf=model_rf.predict_proba(new_X_test)
            f=open('new_results_downsampling_data/'+str(repeats)+'/mi_rf_cv/results_'+str(n_features)+'_'+str(count)+'.txt','w')
            for i in range(len(new_y_test)):
                f.write(str(new_y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
            f.close()  
            count+=1
        model_rf=RandomForestClassifier(max_depth=4, random_state=0).fit(temp_X_train, y_train)
        y_pred_rf=model_rf.predict_proba(temp_X_test)
        f=open('new_results_downsampling_data/'+str(repeats)+'/mi_rf_cv/results_'+str(n_features)+'.txt','w')
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
        f.close()  
    ####SVM
        count=0
        for train_index, test_index in rkf.split(temp_X_train):
            new_X_train=temp_X_train[train_index]
            new_X_test=temp_X_train[test_index]
            new_y_train=y_train[train_index]
            new_y_test=y_train[test_index]
            model_rf=svm.SVC(probability=True).fit(new_X_train, new_y_train)
            y_pred_rf=model_rf.predict_proba(new_X_test)
            f=open('new_results_downsampling_data/'+str(repeats)+'/mi_svm_cv/results_'+str(n_features)+'_'+str(count)+'.txt','w')
            for i in range(len(new_y_test)):
                f.write(str(new_y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
            f.close()  
            count+=1
        model_rf=svm.SVC(probability=True).fit(temp_X_train, y_train)
        y_pred_rf=model_rf.predict_proba(temp_X_test)
        f=open('new_results_downsampling_data/'+str(repeats)+'/mi_svm_cv/results_'+str(n_features)+'.txt','w')
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
        f.close() 
        ####XGB
        count=0
        for train_index, test_index in rkf.split(temp_X_train):
            new_X_train=temp_X_train[train_index]
            new_X_test=temp_X_train[test_index]
            new_y_train=y_train[train_index]
            new_y_test=y_train[test_index]
            model_rf=xgb.XGBClassifier(objective =  'binary:logistic', max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(new_X_train, new_y_train)
            y_pred_rf=model_rf.predict_proba(new_X_test)
            f=open('new_results_downsampling_data/'+str(repeats)+'/mi_xgb_cv/results_'+str(n_features)+'_'+str(count)+'.txt','w')
            for i in range(len(new_y_test)):
                f.write(str(new_y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
            f.close()  
            count+=1
        model_rf=xgb.XGBClassifier(objective =  'binary:logistic', max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(temp_X_train, y_train)
        y_pred_rf=model_rf.predict_proba(temp_X_test)
        f=open('new_results_downsampling_data/'+str(repeats)+'/mi_xgb_cv/results_'+str(n_features)+'.txt','w')
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
        f.close() 
        ###GBR
        count=0
        for train_index, test_index in rkf.split(temp_X_train):
            new_X_train=temp_X_train[train_index]
            new_X_test=temp_X_train[test_index]
            new_y_train=y_train[train_index]
            new_y_test=y_train[test_index]
            model_rf=GradientBoostingClassifier(n_estimators=1000).fit(new_X_train, new_y_train)
            y_pred_rf=model_rf.predict_proba(new_X_test)
            f=open('new_results_downsampling_data/'+str(repeats)+'/mi_gbr_cv/results_'+str(n_features)+'_'+str(count)+'.txt','w')
            for i in range(len(new_y_test)):
                f.write(str(new_y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
            f.close()  
            count+=1
        model_rf=GradientBoostingClassifier(n_estimators=1000).fit(temp_X_train, y_train)
        y_pred_rf=model_rf.predict_proba(temp_X_test)
        f=open('new_results_downsampling_data/'+str(repeats)+'/mi_gbr_cv/results_'+str(n_features)+'.txt','w')
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
        f.close() 
    
        ###ada
        count=0
        for train_index, test_index in rkf.split(temp_X_train):
            new_X_train=temp_X_train[train_index]
            new_X_test=temp_X_train[test_index]
            new_y_train=y_train[train_index]
            new_y_test=y_train[test_index]
            model_rf=AdaBoostClassifier(n_estimators=1000, random_state=0).fit(new_X_train, new_y_train)
            y_pred_rf=model_rf.predict_proba(new_X_test)
            f=open('new_results_downsampling_data/'+str(repeats)+'/mi_ada_cv/results_'+str(n_features)+'_'+str(count)+'.txt','w')
            for i in range(len(new_y_test)):
                f.write(str(new_y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
            f.close()  
            count+=1
        model_rf=AdaBoostClassifier(n_estimators=1000, random_state=0).fit(temp_X_train, y_train)
        y_pred_rf=model_rf.predict_proba(temp_X_test)
        f=open('new_results_downsampling_data/'+str(repeats)+'/mi_ada_cv/results_'+str(n_features)+'.txt','w')
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
        f.close() 
    
        ###LR
        count=0
        for train_index, test_index in rkf.split(temp_X_train):
            new_X_train=temp_X_train[train_index]
            new_X_test=temp_X_train[test_index]
            new_y_train=y_train[train_index]
            new_y_test=y_train[test_index]
            model_rf=LogisticRegression(random_state=0).fit(new_X_train, new_y_train)
            y_pred_rf=model_rf.predict_proba(new_X_test)
            f=open('new_results_downsampling_data/'+str(repeats)+'/mi_lr_cv/results_'+str(n_features)+'_'+str(count)+'.txt','w')
            for i in range(len(new_y_test)):
                f.write(str(new_y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
            f.close()  
            count+=1
        model_rf=LogisticRegression(random_state=0).fit(temp_X_train, y_train)
        y_pred_rf=model_rf.predict_proba(temp_X_test)
        f=open('new_results_downsampling_data/'+str(repeats)+'/mi_lr_cv/results_'+str(n_features)+'.txt','w')
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
        f.close() 

        ###SGD
        count=0
        for train_index, test_index in rkf.split(temp_X_train):
            new_X_train=temp_X_train[train_index]
            new_X_test=temp_X_train[test_index]
            new_y_train=y_train[train_index]
            new_y_test=y_train[test_index]
            model_rf=SGDClassifier(max_iter=1000, tol=1e-3,loss='log').fit(new_X_train, new_y_train)
            y_pred_rf=model_rf.predict_proba(new_X_test)
            f=open('new_results_downsampling_data/'+str(repeats)+'/mi_sgd_cv/results_'+str(n_features)+'_'+str(count)+'.txt','w')
            for i in range(len(new_y_test)):
                f.write(str(new_y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
            f.close()  
            count+=1
        model_rf=SGDClassifier(max_iter=1000, tol=1e-3,loss='log').fit(temp_X_train, y_train)
        y_pred_rf=model_rf.predict_proba(temp_X_test)
        f=open('new_results_downsampling_data/'+str(repeats)+'/mi_sgd_cv/results_'+str(n_features)+'.txt','w')
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
        f.close()     