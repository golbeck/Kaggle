
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


pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/CaterpillarTubePricing'
# dir1='/home/golbeck/Workspace/Kaggle/CaterpillarTubePricing'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

df_train_set=pd.io.parsers.read_table('train_set.csv',sep=',',header=0)
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


n=df_train_set.shape[0]
#dataframe of feature
df=pd.DataFrame()
#fill out with zeros
for i in tube_specs_unique:
    df[i]=np.zeros(n)
#fill out with zeros
for i in comp_type_unique:
    df[i]=np.zeros(n)
#join with train set
df1=pd.concat([df,df_train_set],axis=1)

#generate features from dictionaries then drop other columns


#do not consider at first pass the following data when building features
df_type_component=pd.io.parsers.read_table('type_component.csv',sep=',',header=0)
df_type_connection=pd.io.parsers.read_table('type_connection.csv',sep=',',header=0)
df_tube_end_form=pd.io.parsers.read_table('tube_end_form.csv',sep=',',header=0)
df_type_end_form=pd.io.parsers.read_table('type_end_form.csv',sep=',',header=0)
df_comp_tee=pd.io.parsers.read_table('comp_tee.csv',sep=',',header=0)
df_comp_threaded=pd.io.parsers.read_table('comp_threaded.csv',sep=',',header=0)
df_comp_adaptor=pd.io.parsers.read_table('comp_adaptor.csv',sep=',',header=0)
df_comp_boss=pd.io.parsers.read_table('comp_boss.csv',sep=',',header=0)
df_comp_elbow=pd.io.parsers.read_table('comp_elbow.csv',sep=',',header=0)
df_comp_float=pd.io.parsers.read_table('comp_float.csv',sep=',',header=0)
df_comp_hfl=pd.io.parsers.read_table('comp_hfl.csv',sep=',',header=0)
df_comp_nut=pd.io.parsers.read_table('comp_nut.csv',sep=',',header=0)
df_comp_other=pd.io.parsers.read_table('comp_other.csv',sep=',',header=0)
df_comp_sleeve=pd.io.parsers.read_table('comp_sleeve.csv',sep=',',header=0)
df_comp_sleeve=pd.io.parsers.read_table('comp_sleeve.csv',sep=',',header=0)
df_comp_straight=pd.io.parsers.read_table('comp_straight.csv',sep=',',header=0)

#convert str levels to numerical levels
# df=pd.DataFrame()
# for col in dat.columns[2:]:
#     if type(dat[col].ix[0])==str:
#         df[col]=pd.Categorical(dat[col]).labels
#     else:
#         df[col]=dat[col]

tubes=df_tube['tube_assembly_id'].unique()

for tube in tubes: