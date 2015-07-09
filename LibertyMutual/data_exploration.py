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
dat=pd.io.parsers.read_table('train.csv',sep=',',header=0)

#convert str levels to numerical levels
for col in dat.columns:
    if type(dat[col].ix[0])==str:
        dat[col]=pd.Categorical(dat[col]).labels

#################################################################################
#################################################################################
#visualization
#################################################################################
#################################################################################
%pylab
import matplotlib.pyplot as plt
import matplotlib as mpl

ind=0
for col in dat.columns:
    ind+=1
    plt.figure(ind)
    dat[col].hist()
    plt.suptitle("Column: %s" %col)
    plt.show()



