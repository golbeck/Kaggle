# 136 features, with component types (29 total) in separate columns, 
# with quantities of each as the entries for each tube assembly;
# separate columns also for each tube specification (85)
import os
import sys
import time
import datetime
import operator
from sklearn import ensemble, preprocessing
import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import xgboost as xgb


# pwd_temp=os.getcwd()
# # dir1='/home/sgolbeck/workspace/Kaggle/CaterpillarTubePricing'
# dir1='/home/golbeck/Workspace/Kaggle/CaterpillarTubePricing'
# dir1=dir1+'/data' 
# if pwd_temp!=dir1:
#     os.chdir(dir1)

#training set
df_train_set=pd.io.parsers.read_table('train_set.csv',sep=',',header=0)
Y_train=df_train_set['cost']
df_train_set.drop('cost', axis=1, inplace=True)

#test set
df_test_set=pd.io.parsers.read_table('test_set.csv',sep=',',header=0)

#use for generating features
df_bill_of_materials=pd.io.parsers.read_table('bill_of_materials.csv',sep=',',header=0)
df_components=pd.io.parsers.read_table('components.csv',sep=',',header=0)
df_specs=pd.io.parsers.read_table('specs.csv',sep=',',header=0)
df_tube=pd.io.parsers.read_table('tube.csv',sep=',',header=0)
##########################################################################################
##########################################################################################
#generate dictionary using components csv file. 
#Each component (over 2000) is assigned a type (29 total)
comp_type={}
for ind in range(df_components.shape[0]):
    comp_type[df_components.ix[ind,0]]=df_components.ix[ind,2]

comp_type_unique=list(set(comp_type.values()))
comp_type_unique.sort()
#generate dictionaries using bill of materials csv file. Each tube is assigned 
#its components and their quantities in separate dictionaries.
tube_comp={}
tube_quant={}
for ind in range(df_bill_of_materials.shape[0]):
    ii=1
    temp_comp_id=[]
    temp_quant=[]
    while  isinstance(df_bill_of_materials.ix[ind,ii], str):
        temp_comp_id.append(df_bill_of_materials.ix[ind,ii])
        temp_quant.append(df_bill_of_materials.ix[ind,ii+1])
        ii+=2
        if ii>15:
            break
    tube_comp[df_bill_of_materials.ix[ind,0]]=temp_comp_id
    tube_quant[df_bill_of_materials.ix[ind,0]]=temp_quant
##########################################################################################
##########################################################################################
#generate dictionaries using specs csv file. Each tube is assigned its specifications.
tube_specs={}
for ind in range(df_specs.shape[0]):
    ii=1
    temp_specs=[]
    while  isinstance(df_specs.ix[ind,ii], str):
        temp_specs.append(df_specs.ix[ind,ii])
        ii+=1
        if ii>7:
            break
    tube_specs[df_specs.ix[ind,0]]=temp_specs


#generate list of the unique tube specifications (85 total)
temp=[]
for key in tube_specs.keys():
    if len(tube_specs[key])>0:
        for i in range(len(tube_specs[key])):
            temp.append(tube_specs[key][i])
tube_specs_unique=list(set(temp))
tube_specs_unique.sort()
##########################################################################################
##########################################################################################
df=pd.DataFrame()
n=df_train_set.shape[0]
#fill out with zeros, one column for each component type category
for i in comp_type_unique:
    df[i]=np.zeros(n)
for i in tube_specs_unique:
    df[i]=np.zeros(n)
df_train_set=pd.concat([df_train_set,df],axis=1)

#generate features from dictionaries then drop other columns
for i in range(n):
    #the tube assembly id for the given row (i)
    tube_assembly=df_train_set['tube_assembly_id'][i]
    #enter 1 in appropriate column for each tube spec type
    n_specs=len(tube_specs[tube_assembly])
    if n_specs>0.0:
        for ii in tube_specs[tube_assembly]:
            df_train_set[ii][i]=1.0
    #enter quantity in appropriate column for each component type
    n_comp_types=len(tube_comp[tube_assembly])
    if n_comp_types>0.0:
        ii_temp=0
        for ii in tube_comp[tube_assembly]:
            ii_comp_type=comp_type[ii]
            df_train_set[ii_comp_type][i]+=tube_quant[tube_assembly][ii_temp]
            ii_temp+=1

#do the same for the test set
df=pd.DataFrame()
n=df_test_set.shape[0]
#fill out with zeros, one column for each component type category
for i in comp_type_unique:
    df[i]=np.zeros(n)
for i in tube_specs_unique:
    df[i]=np.zeros(n)

df_test_set=pd.concat([df_test_set,df],axis=1)

#generate features from dictionaries then drop other columns
for i in range(n):
    #the tube assembly id for the given row (i)
    tube_assembly=df_test_set['tube_assembly_id'][i]
    #enter 1 in appropriate column for each tube spec type
    n_specs=len(tube_specs[tube_assembly])
    if n_specs>0.0:
        for ii in tube_specs[tube_assembly]:
            df_test_set[ii][i]=1.0
    #enter quantity in appropriate column for each component type
    n_comp_types=len(tube_comp[tube_assembly])
    if n_comp_types>0.0:
        ii_temp=0
        for ii in tube_comp[tube_assembly]:
            ii_comp_type=comp_type[ii]
            df_test_set[ii_comp_type][i]+=tube_quant[tube_assembly][ii_temp]
            ii_temp+=1
##########################################################################################
##########################################################################################
# #generate a separate encoding for each component type in the bill of materials
# for ind in range(df_bill_of_materials.shape[0]):
#     ii=1
#     while  isinstance(df_bill_of_materials.ix[ind,ii], str):
#         df_bill_of_materials.ix[ind,ii]=comp_type[df_bill_of_materials.ix[ind,ii]]
#         ii+=2
#         if ii>15:
#             break

# df_bill_of_materials.fillna(0, inplace = True)

# for j in [2*i+1 for i in range(8)]:
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(list(df_bill_of_materials[df_bill_of_materials.columns[j]]))
#     df_bill_of_materials.ix[:,j]=lbl.transform(df_bill_of_materials.ix[:,j])

#merge with train and test sets
# df_train_set=pd.merge(df_train_set, df_bill_of_materials, on ='tube_assembly_id')
# df_test_set=pd.merge(df_test_set, df_bill_of_materials, on ='tube_assembly_id')

##########################################################################################
##########################################################################################
#generate labels for each year and month
n=df_train_set.shape[0]
#generate month and year columns
X_time_train=np.zeros((n,3))
for i in range(n):
    X_time_train[i,:]=[int(y) for y in df_train_set['quote_date'][i].split('-')]

#dataframe of features
n_test=df_test_set.shape[0]
#generate month and year columns
X_time_test=np.zeros((n_test,3))
for i in range(n_test):
    X_time_test[i,:]=[int(y) for y in df_test_set['quote_date'][i].split('-')]

for i in range(3):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(X_time_train[:,i]) + list(X_time_test[:,i]))
    X_time_train[:,i] = lbl.transform(X_time_train[:,i])
    X_time_test[:,i] = lbl.transform(X_time_test[:,i])

df_month_train=pd.DataFrame(X_time_train[:,0:2])
df_month_train.columns=['year','month']
df_month_test=pd.DataFrame(X_time_test[:,0:2])
df_month_test.columns=['year','month']
##########################################################################################
##########################################################################################
#generate labels for each supplier
lbl = preprocessing.LabelEncoder()
lbl.fit(list(df_train_set['supplier']) + list(df_test_set['supplier']))
df_train_set['supplier']=lbl.transform(df_train_set['supplier'])
df_test_set['supplier']=lbl.transform(df_test_set['supplier'])
##########################################################################################
##########################################################################################
cols=['end_a_1x','end_a_2x','end_x_1x','end_x_2x','material_id','end_a','end_x']
for i in range(len(cols)):
    df_tube[cols[i]]=pd.Categorical(df_tube[cols[i]]).labels



# df_tube['end_a_1x']=pd.Categorical(df_tube['end_a_1x']).labels
# df_tube['end_a_2x']=pd.Categorical(df_tube['end_a_2x']).labels
# df_tube['end_x_1x']=pd.Categorical(df_tube['end_x_1x']).labels
# df_tube['end_x_2x']=pd.Categorical(df_tube['end_x_2x']).labels

df_train_set=pd.merge(df_train_set, df_tube, on ='tube_assembly_id')
df_test_set=pd.merge(df_test_set, df_tube, on ='tube_assembly_id')

# df_train_set.drop('material_id', axis=1, inplace=True)
# df_train_set.drop('end_a', axis=1, inplace=True)
# df_train_set.drop('end_x', axis=1, inplace=True)
# df_test_set.drop('material_id', axis=1, inplace=True)
# df_test_set.drop('end_a', axis=1, inplace=True)
# df_test_set.drop('end_x', axis=1, inplace=True)

#join with train set
df1=pd.concat([df_train_set,df_month_train],axis=1)

#remove supplier, tube_assembly_id
df1.drop('quote_date', axis=1, inplace=True)
# df1.drop("supplier", axis=1, inplace=True)
# df1.drop("tube_assembly_id", axis=1, inplace=True)
df1['bracket_pricing']=pd.Categorical(df1['bracket_pricing']).labels

del df_train_set, df_month_train, df_tube

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#join with test set
df2=pd.concat([df_test_set,df_month_test],axis=1)

#remove supplier, tube_assembly_id
df2.drop("id", axis=1, inplace=True)
df2.drop('quote_date', axis=1, inplace=True)
# df2.drop("supplier", axis=1, inplace=True)
df2.drop("tube_assembly_id", axis=1, inplace=True)
df2['bracket_pricing']=pd.Categorical(df2['bracket_pricing']).labels

del df_test_set, df_month_test



##########################################################################################
##########################################################################################
##########################################################################################
# include interactions and quadratic term to generate cross-sectional area and tube volume
df1['diameter2']=(df1['diameter']**2.0)
df2['diameter2']=(df2['diameter']**2.0)
df1['length2']=(df1['length']**2.0)
df2['length2']=(df2['length']**2.0)
df1['wall2']=df1['wall']**2.0
df2['wall2']=df2['wall']**2.0
# df1['bend_radius2']=df1['bend_radius']**2.0
# df2['bend_radius2']=df2['bend_radius']**2.0

df1['diam_wall']=df1['diameter']*df1['wall']
df2['diam_wall']=df2['diameter']*df2['wall']
df1['length_diam']=df1['length']*df1['diameter']
df2['length_diam']=df2['length']*df2['diameter']
df1['length_diam2']=df1['length']*(df1['diameter']**2.0)
df2['length_diam2']=df2['length']*(df2['diameter']**2.0)
df1['length_wall']=df1['length']*df1['wall']
df2['length_wall']=df2['length']*df2['wall']

##########################################################################################
##########################################################################################
##########################################################################################
Y_train.to_csv("Y_train.csv",header=True,index=False)
df1.fillna(0, inplace = True)
df2.fillna(0, inplace = True)
df1.to_csv("X_train_143.csv",header=True,index=False)
df2.to_csv("X_test_143.csv",header=True,index=False)
