
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

df_bill_of_materials=pd.io.parsers.read_table('bill_of_materials.csv',sep=',',header=0)
df_comp_adaptor=pd.io.parsers.read_table('comp_adaptor.csv',sep=',',header=0)
df_comp_boss=pd.io.parsers.read_table('comp_boss.csv',sep=',',header=0)
df_comp_elbow=pd.io.parsers.read_table('comp_elbow.csv',sep=',',header=0)
df_comp_float=pd.io.parsers.read_table('comp_float.csv',sep=',',header=0)
df_comp_hfl=pd.io.parsers.read_table('comp_hfl.csv',sep=',',header=0)
df_comp_nut=pd.io.parsers.read_table('comp_nut.csv',sep=',',header=0)
df_components=pd.io.parsers.read_table('components.csv',sep=',',header=0)
df_comp_other=pd.io.parsers.read_table('comp_other.csv',sep=',',header=0)
df_comp_sleeve=pd.io.parsers.read_table('comp_sleeve.csv',sep=',',header=0)
df_comp_sleeve=pd.io.parsers.read_table('comp_sleeve.csv',sep=',',header=0)
df_comp_straight=pd.io.parsers.read_table('comp_straight.csv',sep=',',header=0)
df_comp_tee=pd.io.parsers.read_table('comp_tee.csv',sep=',',header=0)
df_comp_threaded=pd.io.parsers.read_table('comp_threaded.csv',sep=',',header=0)
df_specs=pd.io.parsers.read_table('specs.csv',sep=',',header=0)
df_tube=pd.io.parsers.read_table('tube.csv',sep=',',header=0)
df_tube_end_form=pd.io.parsers.read_table('tube_end_form.csv',sep=',',header=0)
df_type_component=pd.io.parsers.read_table('type_component.csv',sep=',',header=0)
df_type_connection=pd.io.parsers.read_table('type_connection.csv',sep=',',header=0)
df_type_end_form=pd.io.parsers.read_table('type_end_form.csv',sep=',',header=0)


#convert str levels to numerical levels
# df=pd.DataFrame()
# for col in dat.columns[2:]:
#     if type(dat[col].ix[0])==str:
#         df[col]=pd.Categorical(dat[col]).labels
#     else:
#         df[col]=dat[col]

tubes=df_tube['tube_assembly_id'].unique()

for tube in tubes: