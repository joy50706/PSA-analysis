# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:24:54 2022

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
import os
import numpy as np
import pandas as pd
import random
from sklearn.inspection import permutation_importance
dir_name='rfe'

feature_english_name=['sexmale','sexfemale','age','PASI','BSA','height','weight','W','H','WHR','BMI','SBP','DBP',
                      'PASI1A','PASI2A','PASI3A','PASI4A','PASI4E','PASI4D','PASI4I','BSA1','BSA2','BSA3',
                      'BSA4','BSA5','BSA6','BSA7','BSA8','BSA9','BSA10','BSA11','BSA12','BSA13','Blood1normalDecrease',
                      'Blood1normalIncrease','Blood1normalYes','Blood1significanceNo','Blood1significanceYes',
                      'Blood2normalDecrease','Blood2normalIncrease','Blood2normalYes','Blood2significanceNo','Blood2significanceYes',
                      'Blood3normalIncrease','Blood3normalDecrease','Blood3normalYes','Blood3significanceNo','Blood3significanceYes',
                      'Blood4normalIncrease','Blood4normalDecrease','Blood4normalYes','Blood4significanceNo','Blood4significanceYes',
                      'Blood5normalIncrease','Blood5normalDecrease','Blood5normalYes','Blood5significanceNo','Blood5significanceYes',
                      'Blood6normalDecrease','Blood6normalIncrease','Blood6normalYes','Blood6significanceNo','Blood6significanceYes',
                      'urine1testPositive','urine1testNegative','urine1normalNo','urine1normalYes','urine1significanceNo','urine1significanceYes',
                      'urine2testPositive','urine2testNegative','urine2normalNo','urine2normalYes','urine2significanceNo','urine2significanceYes',
                      'urine3testPositive','urine3testNegative','urine3normalNo','urine3normalYes','urine3significanceNo','urine3significanceYes',
                      'urine4testPositive','urine4testNegative','urine4normalNo','urine4normalYes','urine4significanceNo','urine4significanceYes',
                      'Liverkidney1normal0','Liverkidney1normalIncrease','Liverkidney1normalDecrease','Liverkidney1normalYes','Liverkidney1significance0','Liverkidney1significanceNo','Liverkidney1significanceYes',
                      'Liverkidney2normaleIncrease','Liverkidney2normalDecrease','Liverkidney2normalYes','Liverkidney2significanceNo','Liverkidney2significanceYes',
                      'Liverkidney3normal0','Liverkidney3normalIncrease','Liverkidney3normalDecrease','Liverkidney3normalYes','Liverkidney3significance0','Liverkidney3significanceNo','Liverkidney3significanceYes',
                      'Liverkidney4normalIncrease','Liverkidney4normalDecrease','Liverkidney4normalYes','Liverkidney4significanceNo','Liverkidney4significanceYes',
                      'Liverkidney5normalIncrease','Liverkidney5normalDecrease','Liverkidney5normalYes','Liverkidney5significanceNo','Liverkidney5significanceYes',
                      'Liverkidney6normal0','Liverkidney6normalIncrease','Liverkidney6normalYes',
                      'Liverkidney7normal0','Liverkidney7normalIncrease','Liverkidney7normalDecrease','Liverkidney7normalYes',
                      'Liverkidney8normalIncrease','Liverkidney8normalDecrease','Liverkidney8normalYes','Liverkidney8significanceNo','Liverkidney8significanceYes',
                      'Liverkidney9normalIncrease','Liverkidney9normalDecrease','Liverkidney9normalYes','Liverkidney9significanceNo','Liverkidney9significanceYes',
                      'Liverkidney10normalIncrease','Liverkidney10normalDecrease','Liverkidney10normalYes','Liverkidney10significanceNo','Liverkidney10significanceYes',
                      'Bloodfat1normalIncrease','Bloodfat1normalDecrease','Bloodfat1normalYes','Bloodfat1significanceNo','Bloodfat1significanceYes',
                      'Bloodfat2normalIncrease','Bloodfat2normalDecrease','Bloodfat2normalYes','Bloodfat2significanceNo','Bloodfat2significanceYes',
                      'Bloodfat3normalIncrease','Bloodfat3normalDecrease','Bloodfat3normalYes','Bloodfat3significanceNo','Bloodfat3significanceYes',
                      'Bloodfat4normalIncrease','Bloodfat4normalDecrease','Bloodfat4normalYes','Bloodfat4significanceNo','Bloodfat4significanceYes',
                      'Electrolyte1normalDecrease','Electrolyte1normalYes','Electrolyte1significanceNo','Electrolyte1significanceYes',
                      'Electrolyte2normalIncrease','Electrolyte2normalDecrease','Electrolyte2normalYes','Electrolyte2significanceNo','Electrolyte2significanceYes',
                      'Electrolyte3normalIncrease','Electrolyte3normalDecrease','Electrolyte3normalYes','Electrolyte3significanceNo','Electrolyte3significanceYes',
                      '25hydroxyvitaminD31normal0','25hydroxyvitaminD3Increase','25hydroxyvitaminD31normalDecrease','25hydroxyvitaminD31normalYes','25hydroxyvitaminD31significance0','25hydroxyvitaminD31significanceNo','25hydroxyvitaminD31significanceYes',
                      'ESR1normal0','ESR1normalIncrease','ESR1normalYes','ESR1significance0','ESR1significanceNo','ESR1significanceYes',
                      'Rheumatism1normal0','Rheumatism1normalIncrease','Rheumatism1normalYes',
                      'Rheumatism2normal0','Rheumatism2normalIncrease','Rheumatism2normalYes','Rheumatism2significance0','Rheumatism2significanceNo','Rheumatism2significanceYes',
                      'Rheumatism3normal0','Rheumatism3normalIncrease','Rheumatism3normalYes','Rheumatism3significance0','Rheumatism3significanceNo','Rheumatism3significanceYes',
                      'HLAB271test0','HLAB271testPositive','HLAB271testNegative','HLAB271normal0','HLAB271normalNo','HLAB271normalYes','HLAB271significance0','HLAB271significanceNo','HLAB271significanceYes',
                      'DLQI','stageProgression','stage1','stageStationary','stageRegression','seasonCannotRemember','seasonSummer','seasonWinter','seasonAutumn','seasonSpring']

useful_fs_method_df=pd.read_csv('new_results_downsampling_data/useful_fs_method_nfeature.csv')

useful_fs_method=useful_fs_method_df.values.tolist()
for repeats in range(40):
    X_train=np.load(f'new_results_downsampling_data/data/X_train_{repeats}.npy')
    X_test=np.load(f'new_results_downsampling_data/data/X_test_{repeats}.npy')
    
    y_train=np.load(f'new_results_downsampling_data/data/y_train_{repeats}.npy')
    y_test=np.load(f'new_results_downsampling_data/data/y_test_{repeats}.npy')  
    train_df=pd.read_csv(f'new_results_downsampling_data/data/X_train_df_{repeats}.csv')
    for n in range(5):
        
        temp_data=useful_fs_method[repeats*5+n]
        
        if temp_data[2] not in ['lr','sgd']:
            continue
        
        
        if temp_data[1]=='mi':
            selector=SelectKBest(mutual_info_classif, k=temp_data[3]).fit(X_train, y_train)
            selected_index=selector.get_support()
            temp_X_train=selector.transform(X_train)
            selected_english_features=[]
            for i in range(len(feature_english_name)):
                if selected_index[i]==True:
                    selected_english_features.append(feature_english_name[i]) 
        
        elif temp_data[1]=='mrmr':
            selected_features_name=pymrmr.mRMR(train_df,'MIQ', temp_data[3])
            new_index=[]
            selected_english_features=[]
            features_name=list(train_df.columns)[1:]
            for i in range(len(features_name)):
                if features_name[i] in selected_features_name:
                    new_index.append(i)
                    selected_english_features.append(feature_english_name[i])
            temp_X_train=X_train[:,new_index]
     
        elif temp_data[1]=='rfe':
            estimator = RandomForestClassifier(max_depth=4, random_state=0)
            selector = RFE(estimator, n_features_to_select=temp_data[3], step=1)
            selector = selector.fit(X_train, y_train)
            selected_index=selector.support_
            temp_X_train=selector.transform(X_train)
            selected_english_features=[]
            for i in range(len(feature_english_name)):
                if selected_index[i]==True:
                    selected_english_features.append(feature_english_name[i])
   
        elif temp_data[1]=='f':
            selector=SelectKBest(mutual_info_classif, k=temp_data[3]).fit(X_train, y_train)
            selected_index=selector.get_support()
            temp_X_train=selector.transform(X_train)
            selected_english_features=[]
            for i in range(len(feature_english_name)):
                if selected_index[i]==True:
                    selected_english_features.append(feature_english_name[i])            
    
        elif temp_data[1]=='chi2':
            selector=SelectKBest(chi2, k=temp_data[3]).fit(X_train, y_train)
            selected_index=selector.get_support()
            temp_X_train=selector.transform(X_train)
            selected_english_features=[]
            for i in range(len(feature_english_name)):
                if selected_index[i]==True:
                    selected_english_features.append(feature_english_name[i])             
            
        if temp_data[2]=='gbr':
            model_rf=GradientBoostingClassifier(n_estimators=1000).fit(temp_X_train, y_train)
            result = permutation_importance(model_rf, temp_X_train, y_train, n_repeats=10,random_state=0)
            pi_mean=result.importances_mean
            pi_std=result.importances_std        
            save_feature_pi_df=pd.DataFrame({'features':selected_english_features,'piMean':pi_mean,'piStd':pi_std})
            save_feature_pi_df.to_csv(f'new_results_downsampling_data/feature_importance/{temp_data[0]}_{temp_data[1]}_{temp_data[2]}_{temp_data[3]}.csv',index=None)
            
        elif temp_data[2]=='rf':
            model_rf=RandomForestClassifier(max_depth=4, random_state=0).fit(temp_X_train, y_train)
            result = permutation_importance(model_rf, temp_X_train, y_train, n_repeats=10,random_state=0)
            pi_mean=result.importances_mean
            pi_std=result.importances_std
            save_feature_pi_df=pd.DataFrame({'features':selected_english_features,'piMean':pi_mean,'piStd':pi_std})
            save_feature_pi_df.to_csv(f'new_results_downsampling_data/feature_importance/{temp_data[0]}_{temp_data[1]}_{temp_data[2]}_{temp_data[3]}.csv',index=None)

        elif temp_data[2]=='xgb':
            model_rf=xgb.XGBClassifier(objective =  'binary:logistic', max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(temp_X_train, y_train)
            result = permutation_importance(model_rf, temp_X_train, y_train, n_repeats=10,random_state=0)
            pi_mean=result.importances_mean
            pi_std=result.importances_std   
            save_feature_pi_df=pd.DataFrame({'features':selected_english_features,'piMean':pi_mean,'piStd':pi_std})
            save_feature_pi_df.to_csv(f'new_results_downsampling_data/feature_importance/{temp_data[0]}_{temp_data[1]}_{temp_data[2]}_{temp_data[3]}.csv',index=None)

        elif temp_data[2]=='svm':
            model_rf=svm.SVC(probability=True).fit(temp_X_train, y_train)
            result = permutation_importance(model_rf, temp_X_train, y_train, n_repeats=10,random_state=0)
            pi_mean=result.importances_mean
            pi_std=result.importances_std 
            save_feature_pi_df=pd.DataFrame({'features':selected_english_features,'piMean':pi_mean,'piStd':pi_std})
            save_feature_pi_df.to_csv(f'new_results_downsampling_data/feature_importance/{temp_data[0]}_{temp_data[1]}_{temp_data[2]}_{temp_data[3]}.csv',index=None)

            
        elif temp_data[2]=='ada':
            model_rf=AdaBoostClassifier(n_estimators=1000, random_state=0).fit(temp_X_train, y_train)
            result = permutation_importance(model_rf, temp_X_train, y_train, n_repeats=10,random_state=0)
            pi_mean=result.importances_mean
            pi_std=result.importances_std      

            save_feature_pi_df=pd.DataFrame({'features':selected_english_features,'piMean':pi_mean,'piStd':pi_std})
            save_feature_pi_df.to_csv(f'new_results_downsampling_data/feature_importance/{temp_data[0]}_{temp_data[1]}_{temp_data[2]}_{temp_data[3]}.csv',index=None)


        elif temp_data[2]=='sgd':
            model_rf=SGDClassifier(max_iter=1000, tol=1e-3,loss='log').fit(temp_X_train, y_train)
            result = permutation_importance(model_rf, temp_X_train, y_train, n_repeats=10,random_state=0)
            pi_mean=result.importances_mean
            pi_std=result.importances_std  
            save_feature_pi_df=pd.DataFrame({'features':selected_english_features,'piMean':pi_mean,'piStd':pi_std})
            save_feature_pi_df.to_csv(f'new_results_downsampling_data/feature_importance/{temp_data[0]}_{temp_data[1]}_{temp_data[2]}_{temp_data[3]}.csv',index=None)


        elif temp_data[2]=='lr':
            model_rf=LogisticRegression(random_state=0).fit(temp_X_train, y_train)
            result = permutation_importance(model_rf, temp_X_train, y_train, n_repeats=10,random_state=0)
            pi_mean=result.importances_mean
            pi_std=result.importances_std 
            save_feature_pi_df=pd.DataFrame({'features':selected_english_features,'piMean':pi_mean,'piStd':pi_std})
            save_feature_pi_df.to_csv(f'new_results_downsampling_data/feature_importance/{temp_data[0]}_{temp_data[1]}_{temp_data[2]}_{temp_data[3]}.csv',index=None)


#     temp_dir_list=os.listdir('new_results_downsampling_data/'+str(repeats))
#     method_list=['ada','gbr','lr','rf','sgd','svm','xgb']

    #     model_rf=RandomForestClassifier(max_depth=4, random_state=0).fit(temp_X_train, y_train)
    #     y_pred_rf=model_rf.predict_proba(temp_X_test)
    #     f=open('new_results_downsampling_data/'+str(repeats)+'/rfe_rf_cv/results_'+str(n_features)+'.txt','w')
    #     for i in range(len(y_test)):
    #         f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
    #     f.close()  
    # ####SVM

    #     model_rf=svm.SVC(probability=True).fit(temp_X_train, y_train)
    #     y_pred_rf=model_rf.predict_proba(temp_X_test)
    #     f=open('new_results_downsampling_data/'+str(repeats)+'/rfe_svm_cv/results_'+str(n_features)+'.txt','w')
    #     for i in range(len(y_test)):
    #         f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
    #     f.close() 
    #     ####XGB

    #     model_rf=xgb.XGBClassifier(objective =  'binary:logistic', max_depth=5, subsample=0.9,learning_rate=0.1, n_estimators=1000).fit(temp_X_train, y_train)
    #     y_pred_rf=model_rf.predict_proba(temp_X_test)
    #     f=open('new_results_downsampling_data/'+str(repeats)+'/rfe_xgb_cv/results_'+str(n_features)+'.txt','w')
    #     for i in range(len(y_test)):
    #         f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
    #     f.close() 
    #     ###GBR

    #     model_rf=GradientBoostingClassifier(n_estimators=1000).fit(temp_X_train, y_train)
    #     y_pred_rf=model_rf.predict_proba(temp_X_test)
    #     f=open('new_results_downsampling_data/'+str(repeats)+'/rfe_gbr_cv/results_'+str(n_features)+'.txt','w')
    #     for i in range(len(y_test)):
    #         f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
    #     f.close() 
    
    #     ###ada

    #     model_rf=AdaBoostClassifier(n_estimators=1000, random_state=0).fit(temp_X_train, y_train)
    #     y_pred_rf=model_rf.predict_proba(temp_X_test)
    #     f=open('new_results_downsampling_data/'+str(repeats)+'/rfe_ada_cv/results_'+str(n_features)+'.txt','w')
    #     for i in range(len(y_test)):
    #         f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
    #     f.close() 
    
    #     ###LR

    #     model_rf=LogisticRegression(random_state=0).fit(temp_X_train, y_train)
    #     y_pred_rf=model_rf.predict_proba(temp_X_test)
    #     f=open('new_results_downsampling_data/'+str(repeats)+'/rfe_lr_cv/results_'+str(n_features)+'.txt','w')
    #     for i in range(len(y_test)):
    #         f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
    #     f.close() 

    #     ###SGD

    #     model_rf=SGDClassifier(max_iter=1000, tol=1e-3,loss='log').fit(temp_X_train, y_train)
    #     y_pred_rf=model_rf.predict_proba(temp_X_test)
    #     f=open('new_results_downsampling_data/'+str(repeats)+'/rfe_sgd_cv/results_'+str(n_features)+'.txt','w')
    #     for i in range(len(y_test)):
    #         f.write(str(y_test[i])+'\t'+str(y_pred_rf[i][1])+'\n')
    #     f.close()     