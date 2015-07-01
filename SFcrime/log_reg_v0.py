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

dat=pd.io.parsers.read_table('train.csv',sep=',',header=0)
dat.drop('Descript', axis=1, inplace=True)
dat.drop('Resolution', axis=1, inplace=True)
#convert categories to 0-1 indicator matrix
crime_cat=pd.get_dummies(dat['Category'])
weekday_cat=pd.get_dummies(dat['DayOfWeek'])
district_cat=pd.get_dummies(dat['PdDistrict'])
#strip out times
# time_of_day=dat['Dates'].str.split(' ').str[1].str[0:5]
time_of_day=dat['Dates'].str.split(' ').str[1]
def to_seconds(s):
    hr, min, sec = [float(x) for x in s.split(':')]
    return hr*3600 + min*60 + sec
time_of_day=[to_seconds(x) for x in time_of_day]
# time_of_day=pd.to_datetime(time_of_day)
