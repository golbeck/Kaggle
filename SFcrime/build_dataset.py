import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd

pwd_temp=os.getcwd()
# dir1='/home/sgolbeck/workspace/Kaggle/SFcrime'
dir1='/home/golbeck/Workspace/Kaggle/SFcrime'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

#################################################################################
#################################################################################
#train set
#################################################################################
#################################################################################
dat=pd.io.parsers.read_table('train.csv',sep=',',header=0)
dat.drop('Descript', axis=1, inplace=True)
dat.drop('Resolution', axis=1, inplace=True)

#get indices of all outlier Y values
n_90=np.count_nonzero(dat['Y']>40.0)
indices_max=dat['Y'].argsort()[-n_90:]
# dat['Y'][indices_max]
#delete rows with outlider values
dat=dat.drop(indices_max)

#convert categories to 0-1 indicator matrix
# crime_cat=pd.get_dummies(dat['Category'])
weekday_cat=pd.get_dummies(dat['DayOfWeek'])
district_cat=pd.get_dummies(dat['PdDistrict'])
#strip out times
# time_of_day=dat['Dates'].str.split(' ').str[1].str[0:5]
time_of_day=dat['Dates'].str.split(' ').str[1]
def to_seconds(s):
    hr, min, sec = [float(x) for x in s.split(':')]
    return hr*3600 + min*60 + sec
time_of_day=np.asarray([to_seconds(x) for x in time_of_day])
#normalize
time_of_day/=86400.0
# time_of_day=pd.to_datetime(time_of_day)

#normalize X and Y coordinates to [0,1]
y_min=dat['Y'].min()
y_max=dat['Y'].max()
x_min=dat['X'].min()
x_max=dat['X'].max()


dat['Y']-=y_min-0.01
dat['Y']/=(y_max-y_min+0.011)
dat['X']-=x_max+0.01
dat['X']/=(x_min-x_max-0.011)

print dat['X'].max(), dat['X'].min()
print dat['Y'].max(), dat['Y'].min()

#create matrix of features
X_train=np.column_stack((time_of_day,dat['X'],dat['Y'],weekday_cat))
#create output labels array
Y_train=pd.Categorical(dat['Category']).labels
#################################################################################
#################################################################################
#test set
#################################################################################
#################################################################################
dat=pd.io.parsers.read_table('test.csv',sep=',',header=0)

#get indices of all outlier Y values
# n_90=np.count_nonzero(dat['Y']>40.0)
# indices_max=dat['Y'].argsort()[-n_90:]
# #delete rows with outlider values
# dat=dat.drop(indices_max)

#convert categories to 0-1 indicator matrix
# crime_cat=pd.get_dummies(dat['Category'])
weekday_cat=pd.get_dummies(dat['DayOfWeek'])
district_cat=pd.get_dummies(dat['PdDistrict'])
#strip out times
# time_of_day=dat['Dates'].str.split(' ').str[1].str[0:5]
time_of_day=dat['Dates'].str.split(' ').str[1]
def to_seconds(s):
    hr, min, sec = [float(x) for x in s.split(':')]
    return hr*3600 + min*60 + sec
time_of_day=np.asarray([to_seconds(x) for x in time_of_day])
#normalize
time_of_day/=86400.0

#normalize X and Y coordinates to [0,1]
dat['Y']-=y_min-0.01
dat['Y']/=(y_max-y_min+0.011)
dat['X']-=x_max+0.01
dat['X']/=(x_min-x_max-0.011)

print dat['X'].max(), dat['X'].min()
print dat['Y'].max(), dat['Y'].min()

#create matrix of features
X_test=np.column_stack((time_of_day,dat['X'],dat['Y'],weekday_cat))

#################################################################################
#################################################################################
#save data to compressed format
#################################################################################
#################################################################################
#save data to csv files 
np.savetxt("X_train.gz", X_train, delimiter=",")
np.savetxt("Y_train.gz", Y_train, delimiter=",")
np.savetxt("X_test.gz", X_test, delimiter=",")

####load using loadtxt
##        X_train_test=np.loadtxt("X_train.gz",delimiter=",")
####test for equality
##        np.array_equal(X_train,X_train_test)
