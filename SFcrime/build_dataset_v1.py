import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd

pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/SFcrime'
# dir1='/home/golbeck/Workspace/Kaggle/SFcrime'
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
weekday_cat=pd.Categorical(dat['DayOfWeek']).labels
district_cat=pd.Categorical(dat['PdDistrict']).labels
#strip out times
# time_of_day=dat['Dates'].str.split(' ').str[1].str[0:5]
time_of_day=dat['Dates'].str.split(' ').str[1]
def to_seconds(s):
    hr, min, sec = [float(x) for x in s.split(':')]
    return hr*3600 + min*60 + sec
#generate time of day in seconds
time_of_day=np.asarray([to_seconds(x) for x in time_of_day])

#extract month
month_=dat['Dates'].str.split(' ').str[0]
def to_month(s):
    year, month, day = [float(x) for x in s.split('-')]
    return month
#convert month to dummy variable
month_=pd.Series([to_month(x) for x in month_])
month_cat=pd.Categorical(month_).labels
#normalize
time_of_day/=86400.0

#create matrix of features
X_train=np.column_stack((time_of_day,dat['X'],dat['Y'],weekday_cat,district_cat,month_cat))
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
weekday_cat=pd.Categorical(dat['DayOfWeek']).labels
district_cat=pd.Categorical(dat['PdDistrict']).labels
#strip out times
# time_of_day=dat['Dates'].str.split(' ').str[1].str[0:5]
time_of_day=dat['Dates'].str.split(' ').str[1]
time_of_day=np.asarray([to_seconds(x) for x in time_of_day])
#normalize
time_of_day/=86400.0


month_=dat['Dates'].str.split(' ').str[0]
#convert month to dummy variable
month_=pd.Series([to_month(x) for x in month_])
month_cat=pd.Categorical(month_).labels

#create matrix of features
X_test=np.column_stack((time_of_day,dat['X'],dat['Y'],weekday_cat,month_cat))

#################################################################################
#################################################################################
#save data to compressed format
#################################################################################
#################################################################################
#save data to csv files 
np.savetxt("X_train_v1.gz", X_train, delimiter=",")
np.savetxt("Y_train.gz", Y_train, delimiter=",")
np.savetxt("X_test_v1.gz", X_test, delimiter=",")

####load using loadtxt
##        X_train_test=np.loadtxt("X_train.gz",delimiter=",")
####test for equality
##        np.array_equal(X_train,X_train_test)
