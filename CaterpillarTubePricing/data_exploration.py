
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
# dir1='/home/sgolbeck/workspace/Kaggle/CaterpillarTubePricing'
dir1='/home/golbeck/Workspace/Kaggle/CaterpillarTubePricing'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

dat=pd.io.parsers.read_table('train_set.csv',sep=',',header=0)

#convert str levels to numerical levels
df=pd.DataFrame()
for col in dat.columns[2:]:
    if type(dat[col].ix[0])==str:
        df[col]=pd.Categorical(dat[col]).labels
    else:
        df[col]=dat[col]
