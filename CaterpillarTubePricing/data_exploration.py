
import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import xgboost as xgb


# pwd_temp=os.getcwd()
# # dir1='/home/sgolbeck/workspace/Kaggle/CaterpillarTubePricing'
# # dir1='/home/golbeck/Workspace/Kaggle/CaterpillarTubePricing'
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

#dataframe of features
n=df_train_set.shape[0]
#generate month and year columns
X_time=np.zeros((n,3))
for i in range(n):
    X_time[i,:]=[int(y) for y in df_train_set['quote_date'][i].split('-')]

cols_year=[str(x) for x in list(set(X_time[:,0]))]
cols_month=[str(x) for x in list(set(X_time[:,1]))]

from statsmodels.tools import categorical
df_year=pd.DataFrame(categorical(X_time[:,0],drop=True),columns=cols_year)
df_month=pd.DataFrame(categorical(X_time[:,1],drop=True),columns=cols_month)

df_tube['end_a_1x']=pd.Categorical(df_tube['end_a_1x']).labels
df_tube['end_a_2x']=pd.Categorical(df_tube['end_a_2x']).labels
df_tube['end_x_1x']=pd.Categorical(df_tube['end_x_1x']).labels
df_tube['end_x_2x']=pd.Categorical(df_tube['end_x_2x']).labels

df_train_set=pd.merge(df_train_set, df_tube, on ='tube_assembly_id')

df_train_set.drop('material_id', axis=1, inplace=True)
df_train_set.drop('end_a', axis=1, inplace=True)
df_train_set.drop('end_x', axis=1, inplace=True)

df=pd.DataFrame()
#fill out with zeros, one column for each specs category
for i in tube_specs_unique:
    df[i]=np.zeros(n)
#fill out with zeros, one column for each component type category
for i in comp_type_unique:
    df[i]=np.zeros(n)

# #fill out with zeros, one column for each supplier
# supplier_list=list(set(df_train_set['supplier']))
# supplier_list_test=list(set(df_test_set['supplier']))
# supplier_list.extend(supplier_list_test)
# supplier_list.sort()
# for i in supplier_list:
#     df[i]=np.zeros(n)

#join with train set
df1=pd.concat([df,df_train_set,df_year,df_month],axis=1)

#generate features from dictionaries then drop other columns
for i in range(n):
    #the tube assembly id for the given row (i)
    tube_assembly=df1['tube_assembly_id'][i]
    #enter 1 in appropriate column for each tube spec type
    n_specs=len(tube_specs[tube_assembly])
    if n_specs>0.0:
        for ii in tube_specs[tube_assembly]:
            df1[ii][i]=1.0
    #enter quantity in appropriate column for each component type
    n_comp_types=len(tube_comp[tube_assembly])
    if n_comp_types>0.0:
        ii_temp=0
        for ii in tube_comp[tube_assembly]:
            ii_comp_type=comp_type[ii]
            df1[ii_comp_type][i]+=tube_quant[tube_assembly][ii_temp]
            ii_temp+=1
    # #enter 1 in appropriate supplier column
    # supplier_label=df1['supplier'][i]
    # df1[supplier_label][i]=1.0

#remove supplier, tube_assembly_id
df1.drop('quote_date', axis=1, inplace=True)
df1.drop("supplier", axis=1, inplace=True)
df1.drop("tube_assembly_id", axis=1, inplace=True)
df1['bracket_pricing']=pd.Categorical(df1['bracket_pricing']).labels

del df, df_train_set, df_year, df_month

# #do not consider at first pass the following data when building features
# df_type_component=pd.io.parsers.read_table('type_component.csv',sep=',',header=0)
# df_type_connection=pd.io.parsers.read_table('type_connection.csv',sep=',',header=0)
# df_tube_end_form=pd.io.parsers.read_table('tube_end_form.csv',sep=',',header=0)
# df_type_end_form=pd.io.parsers.read_table('type_end_form.csv',sep=',',header=0)
# df_comp_tee=pd.io.parsers.read_table('comp_tee.csv',sep=',',header=0)
# df_comp_threaded=pd.io.parsers.read_table('comp_threaded.csv',sep=',',header=0)
# df_comp_adaptor=pd.io.parsers.read_table('comp_adaptor.csv',sep=',',header=0)
# df_comp_boss=pd.io.parsers.read_table('comp_boss.csv',sep=',',header=0)
# df_comp_elbow=pd.io.parsers.read_table('comp_elbow.csv',sep=',',header=0)
# df_comp_float=pd.io.parsers.read_table('comp_float.csv',sep=',',header=0)
# df_comp_hfl=pd.io.parsers.read_table('comp_hfl.csv',sep=',',header=0)
# df_comp_nut=pd.io.parsers.read_table('comp_nut.csv',sep=',',header=0)
# df_comp_other=pd.io.parsers.read_table('comp_other.csv',sep=',',header=0)
# df_comp_sleeve=pd.io.parsers.read_table('comp_sleeve.csv',sep=',',header=0)
# df_comp_sleeve=pd.io.parsers.read_table('comp_sleeve.csv',sep=',',header=0)
# df_comp_straight=pd.io.parsers.read_table('comp_straight.csv',sep=',',header=0)


#dataframe of features
n=df_test_set.shape[0]
#generate month and year columns
X_time=np.zeros((n,3))
for i in range(n):
    X_time[i,:]=[int(y) for y in df_test_set['quote_date'][i].split('-')]

from statsmodels.tools import categorical
df_year=pd.DataFrame(categorical(X_time[:,0],drop=True),columns=list(set(X_time[:,0])))
df_year=pd.concat([pd.DataFrame(np.zeros(n),columns=[1982]),df_year],axis=1)
df_month=pd.DataFrame(categorical(X_time[:,1],drop=True),columns=cols_month)


df_test_set=pd.merge(df_test_set, df_tube, on ='tube_assembly_id')

df_test_set.drop('material_id', axis=1, inplace=True)
df_test_set.drop('end_a', axis=1, inplace=True)
df_test_set.drop('end_x', axis=1, inplace=True)

#dataframe of features
df=pd.DataFrame()
#fill out with zeros, one column for each specs category
for i in tube_specs_unique:
    df[i]=np.zeros(n)
#fill out with zeros, one column for each component type category
for i in comp_type_unique:
    df[i]=np.zeros(n)

#join with test set
df2=pd.concat([df,df_test_set,df_year,df_month],axis=1)

#generate features from dictionaries then drop other columns
for i in range(n):
    #the tube assembly id for the given row (i)
    tube_assembly=df2['tube_assembly_id'][i]
    #enter 1 in appropriate column for each tube spec type
    n_specs=len(tube_specs[tube_assembly])
    if n_specs>0.0:
        for ii in tube_specs[tube_assembly]:
            df2[ii][i]=1.0
    #enter quantity in appropriate column for each component type
    n_comp_types=len(tube_comp[tube_assembly])
    if n_comp_types>0.0:
        ii_temp=0
        for ii in tube_comp[tube_assembly]:
            ii_comp_type=comp_type[ii]
            df2[ii_comp_type][i]+=tube_quant[tube_assembly][ii_temp]
            ii_temp+=1
    # #enter 1 in appropriate supplier column
    # supplier_label=df2['supplier'][i]
    # df2[supplier_label][i]=1.0


#remove supplier, tube_assembly_id
df2.drop("id", axis=1, inplace=True)
df2.drop('quote_date', axis=1, inplace=True)
df2.drop("supplier", axis=1, inplace=True)
df2.drop("tube_assembly_id", axis=1, inplace=True)
df2['bracket_pricing']=pd.Categorical(df2['bracket_pricing']).labels

del df, df_test_set, df_year, df_month

Y_train.to_csv("Y_train.csv",header=True,index=False)
df1.to_csv("X_train.csv",header=True,index=False)
df2.to_csv("X_test.csv",header=True,index=False)