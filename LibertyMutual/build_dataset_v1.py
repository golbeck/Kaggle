# builds dataset, ignoring categorical variables. All continuous variables are transformed
# into principal components after demeaning and standardization
import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing

pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
# dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
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
scaler_list=[]
for col in dat.columns[2:]:
    if type(dat[col].ix[0])!=str:
        #normalize to [0,1]
        df[col] = preprocessing.scale(np.array(dat[col].astype('float')))
        scaler_list.append(preprocessing.StandardScaler().fit(np.array(dat[col].astype('float'))))

n=df.shape[1]
pca = PCA(n_components=n)
pca_fit=pca.fit(df)
df_fit=pca_fit.transform(df)

np.savetxt("X_train_01_v1.gz", df, delimiter=",")
np.savetxt("X_train_pca_v1.gz", df_fit, delimiter=",")
del dat, df
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
ind=0
for col in dat.columns[1:]:
    if type(dat[col].ix[0])!=str:
        #normalize to [0,1]
        df[col] = scaler_list[ind].transform(np.array(dat[col].astype('float')))
        ind+=1

df_fit=pca_fit.transform(df)
np.savetxt("X_test_01_v1.gz", df, delimiter=",")
np.savetxt("X_test_pca_v1.gz", df_fit, delimiter=",")
del dat, df