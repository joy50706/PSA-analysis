# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:52:32 2022

@author: lenovo
"""






import numpy as np
import pandas as pd
import numpy as np
import os
import xlrd
import xlwt
import copy
import xlutils
import datetime
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

book = xlrd.open_workbook('data/cure_test_updated_in_english.xlsx')
sheet = book.sheet_by_index(0)
rows = sheet.nrows
cols = sheet.ncols

check_index=-1

data_dict={}
for i in range(1,rows):
    temp=sheet.cell_value(i,0)
    temp_name=temp.split('+')[0]
    temp_medicine=sheet.cell_value(i,3)
    if temp_name not in data_dict.keys():
        data_dict[temp_name]=[]
    data_dict[temp_name].append(temp_medicine)

for each_key in data_dict.keys():
    data_dict[each_key]=list(set(data_dict[each_key]))
    
selected_names={}
for each_key in data_dict.keys():
    if len(data_dict[each_key])==1 and data_dict[each_key][0]!='' and data_dict[each_key][0]!='Chinese' \
        and data_dict[each_key][0]!='other':
            selected_names[each_key]=data_dict[each_key][0]

data_time_dict={}
for i in range(1,rows):
    temp=sheet.cell_value(i,0)
    temp_name=temp.split('+')[0]
    if temp.split('+')[1]=='00000000':
        continue
    if temp_name not in data_time_dict.keys():
        data_time_dict[temp_name]=[]
    if len(sheet.cell_value(i,4))>0 and is_number(sheet.cell_value(i,4)):
        data_time_dict[temp_name].append([temp.split('+')[1],float(sheet.cell_value(i,4))])

for each_key in data_time_dict.keys():
    temp=data_time_dict[each_key]
    data_time_dict[each_key]=sorted(temp, key=(lambda x: x[0]))

new_data_timediff_dict={}
for each_key in data_time_dict.keys():
    new_data_timediff_dict[each_key]=[]
    temp=data_time_dict[each_key]
    for i in range(len(temp)):
        if i==0:
            new_data_timediff_dict[each_key].append([0,temp[i][1]])
        else:
            d1 = datetime.datetime.strptime(temp[0][0][:4]+'-'+temp[0][0][4:6]+'-'\
                                    +temp[0][0][6:8]+' 00:00:00', '%Y-%m-%d %H:%M:%S')
            d2 = datetime.datetime.strptime(temp[i][0][:4]+'-'+temp[i][0][4:6]+'-'\
                                    +temp[i][0][6:8]+' 00:00:00', '%Y-%m-%d %H:%M:%S')                
            delta_days=(d2-d1).days
            new_data_timediff_dict[each_key].append([delta_days,temp[i][1]])
        
medicine_cluster_dict={}
for each_name in selected_names.keys():
    temp=new_data_timediff_dict[each_name]
    temp_med=selected_names[each_name]
    if temp_med not in medicine_cluster_dict.keys():
        medicine_cluster_dict[temp_med]=[]
    new_temp=[]
    for i in range(1,len(temp)):
        if i==1:
            if temp[i][0]-temp[i-1][0]<=45:
                new_temp.append(temp[i-1])
                new_temp.append(temp[i])
            else:
                break
        else:
            if temp[i][0]-temp[i-1][0]<=45:
                new_temp.append(temp[i])
                
            else:
                break 
    if len(new_temp)>0:
        medicine_cluster_dict[temp_med].append(new_temp)


import copy

old_medicine_cluster_dict=copy.deepcopy(medicine_cluster_dict)




######no less than 3
# temp=old_medicine_cluster_dict['MTX']
# del_temp=[]
# for each_temp in temp:
#     if each_temp[-1][0]>100:
#         del_temp=each_temp
#         break

# old_medicine_cluster_dict['MTX'].remove(del_temp)
# temp=old_medicine_cluster_dict['acitretin']
# del_temp=[]
# for each_temp in temp:
#     if each_temp[-1][0]>67:
#         del_temp=each_temp
#         break

# old_medicine_cluster_dict['acitretin'].remove(del_temp)

#######no less than 10
temp=old_medicine_cluster_dict['IL17']
del_temp=[]
for each_temp in temp:
    if each_temp[-1][0]>100:
        del_temp=each_temp
        break

old_medicine_cluster_dict['IL17'].remove(del_temp)
temp=old_medicine_cluster_dict['acitretin']
del_temp=[]
for each_temp in temp:
    if each_temp[-1][0]>94:
        del_temp=each_temp
        break




old_medicine_cluster_dict['acitretin'].remove(del_temp)





medicine_cluster_dict={}
medicine_clusters_combined_dict={}
for each_key in old_medicine_cluster_dict.keys():
    temp=[]
    temp_list=old_medicine_cluster_dict[each_key]
    medicine_cluster_dict[each_key]=[]
    for each_list in temp_list:
        if each_list[0][1]>=10 :
            temp+=each_list
            medicine_cluster_dict[each_key].append(each_list)
            
    medicine_clusters_combined_dict[each_key]=temp

from matplotlib import pyplot
import matplotlib.pyplot as plt

from pygam import LinearGAM


medicine_color_one=['#FAB406','#FA5D38','#92E834','#1DAEFF','#E549E9','#FD3594','#EEE912','#46D68E']
medicine_color_two=['#D69A04','#E03006','#66B014','#008AD6','#B717BB','#D80268','#BEBA0E','#26AC69']


medicine_list=list(medicine_cluster_dict.keys())
medicine_list=sorted(medicine_list)
# fig = plt.figure(figsize=(15,6))



# i=0
# ax = fig.add_subplot(2, 3, 1)
# for n in range(len(medicine_list)):
#     each_key=medicine_list[n]
#     # if each_key not in ['Corticosteroid','MTX','acitretin']:
#     if each_key not in ['Corticosteroid','MTX','acitretin','TNF','IL17']:
#     # if each_key not in ['Corticosteroid','MTX','acitretin','TNF','IL17']:
#         continue
#     temp=medicine_cluster_dict[each_key]
#     temp_data=medicine_clusters_combined_dict[each_key]
#     if len(temp_data)==0:
#         continue
#     temp_data=np.array(temp_data)
#     X=temp_data[:,0:1]
#     y=np.log2(temp_data[:,1]+1)
#     gam = LinearGAM(n_splines=4).gridsearch(X, y)
#     XX = gam.generate_X_grid(term=0, n=5)
#     # if i==0:


        
#     plt.plot(XX, gam.predict(XX), color=medicine_color_two[n],linewidth=5)
#     plt.title(each_key)
#     plt.ylim(0, 5.5)
#     plt.xlim(0, 80)
#     i+=1
# #34FEBF
# # plt.legend()
# plt.savefig('final_fig/all_medicine_PASI_log_45d_add_all_GAMs_nolessthan10.pdf',dpi=600)
# plt.show()


