import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd

pwd_temp=os.getcwd()
# dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

#################################################################################
#################################################################################
#train set
#################################################################################
#################################################################################
dat=pd.io.parsers.read_table('train.csv',sep=',',header=0)

#convert str levels to numerical levels
df=pd.DataFrame()
for col in dat.columns[2:]:
    if type(dat[col].ix[0])==str:
        df[col]=pd.Categorical(dat[col]).labels
    else:
        df[col]=dat[col]

np.savetxt("Y_train.gz", dat['Hazard'], delimiter=",")
del dat
np.savetxt("X_train_v1.gz", df, delimiter=",")
del df
#################################################################################
#################################################################################
#test set
#################################################################################
#################################################################################
dat=pd.io.parsers.read_table('test.csv',sep=',',header=0)

indices=dat['Id']
np.savetxt("X_test_indices.gz", indices, delimiter=",")
#convert str levels to numerical levels
df=pd.DataFrame()
for col in dat.columns[1:]:
    if type(dat[col].ix[0])==str:
        df[col]=pd.Categorical(dat[col]).labels
    else:
        df[col]=dat[col]

del dat
np.savetxt("X_test_v1.gz", df, delimiter=",")