# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 06:53:47 2022

@author: lenovo
"""







import numpy as np
import pandas as pd
import numpy as np
import os
import xlrd
# import xlwt
import copy
# import xlutils

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

book = xlrd.open_workbook('data/original_test_transformed_selected_20_risk_data_new_data.xlsx')
sheet = book.sheet_by_index(0)
rows = sheet.nrows
cols = sheet.ncols


features_name=[]
for j in range(1,cols-1):
    features_name.append(sheet.cell(0,j).value)
        
labels=[]
for i in range(1,rows):
    labels.append(int(sheet.cell(i,cols-1).value))
data=[]
name_list=[]
for i in range(1,rows):
    name_list.append(sheet.cell(i,0).value)
    temp=[]
    for j in range(1,cols-1):
        if is_number(sheet.cell(i,j).value):
            temp.append(float(sheet.cell(i,j).value))
        else:
            temp.append(sheet.cell(i,j).value)
    data.append(temp)


feature_dict={}
for j in range(len(features_name)):
    feature_dict[features_name[j]]=[]
    for i in range(len(data)):
        feature_dict[features_name[j]].append(data[i][j])

for each_key in feature_dict.keys():
    feature_dict[each_key]=list(set(feature_dict[each_key]))

number_index=list(range(1,32))+[100]
transform_feature_dict={}
for j in range(len(features_name)):
    if j not in number_index:
        temp=feature_dict[features_name[j]]
        temp_dict={}
        for k in range(len(temp)):
            temp_numbers=np.zeros(len(temp),dtype=int)
            temp_numbers[k]=1
            temp_dict[temp[k]]=temp_numbers
        transform_feature_dict[j]=temp_dict
delete_features=['职业']
new_data=[]

for i in range(len(data)):
    temp=[]
    new_features_name=[]
    for j in range(len(data[i])):
        if features_name[j] not in delete_features:
            if j in number_index:
                if data[i][j]=='' or data[i][j]=='-' or data[i][j]=='/':
                    temp.append(np.nan)
                    
                else:
                    temp.append(data[i][j])
                new_features_name.append(features_name[j])
            else:
                temp+=transform_feature_dict[j][data[i][j]].tolist()
                for k in range(len(transform_feature_dict[j][data[i][j]].tolist())):
                    new_features_name.append(features_name[j]+'_'+str(k))
    new_data.append(temp)

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(new_data)
new_data=imp_mean.transform(new_data)

labels=np.array(labels)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pymrmr
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
# import pymrmr
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2,mutual_info_classif,f_classif
# train_df = pd.DataFrame(X_train, columns = features_name)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SequentialFeatureSelector, RFE

label_for_df=labels.reshape(-1,1)
new_data_for_df=np.hstack((label_for_df,new_data))
data_df = pd.DataFrame(new_data_for_df, columns = ['label']+new_features_name)

for n_features in range(20,31,1):
    # selector=SelectKBest( chi2, k=n_features).fit(new_data, labels)
    # selected_index=selector.get_support
    # new_new_data=selector.transform(new_data)

    selected_features_name=pymrmr.mRMR(data_df,'MIQ', n_features)
    new_index=[]
    for i in range(len(new_features_name)):
        if new_features_name[i] in selected_features_name:
            new_index.append(i)
    new_new_data=new_data[:,new_index]

    # estimator = RandomForestClassifier(max_depth=5, random_state=0)
    # selector = RFE(estimator, n_features_to_select=n_features)
    # selector = selector.fit(new_data,  labels)
    # # selected_index=selector.get_support
    # new_new_data=selector.transform(new_data)

 
    loo = LeaveOneOut()
    svm_pred_prob=[]
    rf_pred_prob=[]
    xgb_pred_prob=[]
    ada_pred_prob=[]
    sgd_pred_prob=[]
    gbr_pred_prob=[]
    lr_pred_prob=[]
    true_labels=[]

    for train, test in loo.split(new_new_data):
        X_train=new_new_data[train]
        X_test=new_new_data[test]
        y_train=labels[train]
        y_test=labels[test]

        true_labels.append(y_test[0])
        model_svm=svm.SVC(probability=True).fit(X_train, y_train)
        y_pred_svm=model_svm.predict_proba(X_test)
        svm_pred_prob.append(y_pred_svm[0,1])
    
        model_rf=RandomForestClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
        y_pred_rf=model_rf.predict_proba(X_test)
        rf_pred_prob.append(y_pred_rf[0,1])
    
        model_gbr=GradientBoostingClassifier(n_estimators=1000).fit(X_train, y_train)
        y_pred_gbr=model_gbr.predict_proba(X_test)
        gbr_pred_prob.append(y_pred_gbr[0,1])
 
        model_xgb = xgb.XGBClassifier(objective =  'binary:logistic', max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(X_train, y_train)
        y_pred_xgb = model_xgb.predict_proba(X_test)    
        xgb_pred_prob.append(y_pred_xgb[0,1])
    
        model_lr= LogisticRegression(random_state=0).fit(X_train, y_train)
        y_pred_lr=model_lr.predict_proba(X_test)
        lr_pred_prob.append(y_pred_lr[0,1])

        model_sgd= SGDClassifier(max_iter=1000, tol=1e-3,loss='log').fit(X_train, y_train)
        y_pred_sgd=model_sgd.predict_proba(X_test)
        sgd_pred_prob.append(y_pred_sgd[0,1])
    
        model_ada= AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
        y_pred_ada=model_ada.predict_proba(X_test)  
        ada_pred_prob.append(y_pred_ada[0,1])
    

   
    svm_auc=roc_auc_score(true_labels,svm_pred_prob)
    rf_auc=roc_auc_score(true_labels,rf_pred_prob)
    xgb_auc=roc_auc_score(true_labels,xgb_pred_prob)
    ada_auc=roc_auc_score(true_labels,ada_pred_prob)
    sgd_auc=roc_auc_score(true_labels,sgd_pred_prob)
    gbr_auc=roc_auc_score(true_labels,gbr_pred_prob)
    lr_auc=roc_auc_score(true_labels,lr_pred_prob)
    print(n_features)
    print(f'ada auc:{ada_auc:.4f}, rf_auc:{rf_auc:.4f},xgb_auc:{xgb_auc:.4f}')
    print(f'svm auc:{svm_auc:.4f}, gbr auc:{gbr_auc:.4f}, sgd auc:{sgd_auc:.4f}')
    print(f'lr auc:{lr_auc:.4f}')
    df_save=pd.DataFrame({'true':true_labels,'svm':svm_pred_prob,'rf':rf_pred_prob,'lr':lr_pred_prob,\
                      'xgb':xgb_pred_prob,'ada':ada_pred_prob,'sgd':sgd_pred_prob,'gbr':gbr_pred_prob})
    df_save.to_csv(f'new_results/risk_prediction/new_data_mrmr_short_range_features_{n_features}.csv',index=None)