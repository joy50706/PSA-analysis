# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:49:03 2022

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
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pylab as plt

label_for_df=labels.reshape(-1,1)
new_data_for_df=np.hstack((label_for_df,new_data))
data_df = pd.DataFrame(new_data_for_df, columns = ['label']+new_features_name)



import eli5
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import shap


for n_features in range(27,28,1):
    # selector=SelectKBest( f_classif, k=n_features).fit(new_data, labels)
    # # selected_index=selector.get_support
    # new_new_data=selector.transform(new_data)

    selected_features_name=pymrmr.mRMR(data_df,'MIQ', n_features)
    new_index=[]
    for i in range(len(new_features_name)):
        if new_features_name[i] in selected_features_name:
            new_index.append(i)
    new_new_data=new_data[:,new_index]

    # estimator = RandomForestClassifier(max_depth=5, random_state=0)
    # selector = SequentialFeatureSelector(estimator, n_features_to_select=n_features)
    # # selector = RFE(estimator, n_features_to_select=n_features)
    # selector = selector.fit(new_data,  labels)
    # selected_index=selector.get_support()
    # selected_feature_names=[]
    # for i in range(len(new_features_name)):
    #     if selected_index[i]==True:
    #         selected_feature_names.append(new_features_name[i])

    # new_new_data=selector.transform(new_data)
    
    
    X_train, X_test, y_train, y_test = train_test_split(new_new_data, labels, test_size=0.2, random_state=42)
    # model_ada= AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
    model_xgb =  AdaBoostClassifier(n_estimators=100, random_state=0).fit(new_new_data, labels)
    # perm = PermutationImportance(model_ada,cv=132).fit(new_new_data, labels)
    # eli5.show_weights(perm)
    # model_ada_all_data= AdaBoostClassifier(n_estimators=100, random_state=0).fit(new_new_data, labels)

    result = permutation_importance(model_xgb, new_new_data, labels, n_repeats=100,random_state=0)
    pi_mean=result.importances_mean
    pi_std=result.importances_std  
    selected_features_name[0]='Age'
    selected_features_name[1]='bloodRT-2-Isnormal'
    selected_features_name[2]='ESR-1-IsNormal'
    selected_features_name[3]='LiverRenal-6-IsNormal'
    selected_features_name[4]='IllnessStage'
    selected_features_name[5]='LiverRenal-9-IsNormal'
    selected_features_name[6]='BloodLipid-3-IsNormal'
    selected_features_name[7]='ESR-1-CS'
    selected_features_name[8]='Rheumatism-2-Isnormal'
    selected_features_name[9]='LiverRenal-3-IsNormal'
    selected_features_name[10]='UrineRT-3-IsNormal'
    selected_features_name[11]='LiverRenal-9-IsNormal'
    selected_features_name[12]='BSA-3'
    selected_features_name[13]='LiverRenal-10-CS'
    selected_features_name[14]='bloodRT-2-Isnormal'
    selected_features_name[15]='25HydroxyvitaminD3-CS'
    selected_features_name[16]='LiverRenal-1-IsNormal'
    selected_features_name[17]='BSA-13'
    selected_features_name[18]='UrineRT-4'
    selected_features_name[19]='HLA-B27-1-CS'
    selected_features_name[20]='bloodRT-3-Isnormal'
    selected_features_name[21]='LiverRenal-1-IsNormal'
    selected_features_name[22]='PASI-4'
    selected_features_name[23]='UrineRT-3-IsNormal'
    selected_features_name[24]='BloodLipid-1-CS'
    selected_features_name[25]='UrineRT-4'
    selected_features_name[26]='Rheumatism-4-CS'
    # selected_features_name[12]='25HydroxyvitaminD3Isnormal'
    save_feature_pi_df=pd.DataFrame({'features':selected_features_name,'PImean':pi_mean,'PIstd':pi_std})
    # save_feature_pi_df.to_csv(f'final_fig/new_data_risk_prediction_mrmr_adaboost_feature_importance.csv',index=None)

    explainer = shap.Explainer(model_xgb.predict,new_new_data)
    new_new_data_df=pd.DataFrame(new_new_data,columns=selected_features_name)
    shap_values = explainer(new_new_data)
    shap.summary_plot(shap_values[:,[0,1,2,19,22,21,4,3,23,8]], new_new_data[:,[0,1,2,19,22,21,4,3,23,8]],show=False)
    # plt.savefig('final_fig/new_risk_data_shap.pdf',dpi=600, bbox_inches='tight')
