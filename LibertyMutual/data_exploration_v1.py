import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd

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
df=pd.io.parsers.read_table('train.csv',sep=',',header=0)

#convert str levels to numerical levels
dat=pd.DataFrame()
for col in df.columns[2:]:
    if type(df[col].ix[0])!=str:
        dat[col]=df[col]
    else:
        dat[col]=pd.Categorical(df[col]).labels

# dat.drop('T2_V8',axis=1,inplace=True)
dat.drop('T2_V12',axis=1,inplace=True)
# dat.drop('T1_V16',axis=1,inplace=True)

dat_mean=dat.mean()
dat_std=dat.std()

n_outliers_H=np.zeros(dat.shape[1])
n_outliers_L=np.zeros(dat.shape[1])
ind=0
n_std=4
for col in dat.columns:
    n_outliers_H[ind]=np.count_nonzero(dat[col]>dat_mean[col]+n_std*dat_std[col])
    n_outliers_L[ind]=np.count_nonzero(dat[col]<dat_mean[col]-n_std*dat_std[col])
    ind+=1
    # indices_max=dat['Y'].argsort()[-n:]

n=np.count_nonzero(dat['T2_V2']>36)
indices=dat['T2_V2'].argsort()[-n:]
#################################################################################
#################################################################################
#visualization
#################################################################################
#################################################################################
%pylab
import matplotlib.pyplot as plt
import matplotlib as mpl

#plot all of the histograms
dat.hist()

#################################################################################
#################################################################################
#test set
#################################################################################
#################################################################################
df=pd.io.parsers.read_table('test.csv',sep=',',header=0)

#convert str levels to numerical levels
dat=pd.DataFrame()
for col in df.columns[1:]:
    if type(df[col].ix[0])!=str:
        # dat[col]=pd.Categorical(df[col]).labels
        dat[col]=df[col]

# dat_mean=dat.mean()
# dat_std=dat.std()

n_outliers_H=np.zeros(dat.shape[1])
n_outliers_L=np.zeros(dat.shape[1])
ind=0
n_std=4
for col in dat.columns:
    n_outliers_H[ind]=np.count_nonzero(dat[col]>dat_mean[col]+n_std*dat_std[col])
    n_outliers_L[ind]=np.count_nonzero(dat[col]<dat_mean[col]-n_std*dat_std[col])
    ind+=1
    # indices_max=dat['Y'].argsort()[-n:]
#################################################################################
#################################################################################
#visualization
#################################################################################
#################################################################################
%pylab
import matplotlib.pyplot as plt
import matplotlib as mpl

#plot all of the histograms
dat.hist()



#################################################################################
#################################################################################
%pylab
import matplotlib.pyplot as plt
import matplotlib as mpl

pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
# dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

dat=pd.io.parsers.read_table('train.csv',sep=',',header=0)

#convert str levels to numerical levels
df=pd.DataFrame()
for col in dat.columns[2:]:
    if type(dat[col].ix[0])==str:
        df[col]=pd.Categorical(dat[col]).labels
    else:
        df[col]=dat[col]

df['Hazard']=dat['Hazard']
col_names=df.columns


ind=0
for i in range(len(col_names)-1):
    ind+=1
    plt.figure(ind)
    plt.scatter(df[col_names[i]], df['Hazard'])
    plt.show()
