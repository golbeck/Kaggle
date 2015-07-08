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


time_of_day=dat['Dates'].str.split(' ').str[1]
def to_seconds(s):
    hr, min, sec = [float(x) for x in s.split(':')]
    return hr*3600 + min*60 + sec
time_of_day=np.asarray([to_seconds(x) for x in time_of_day])

weekday_cat=pd.get_dummies(dat['DayOfWeek'])
cols=['Time']+list(weekday_cat.columns)
crime_cat_=pd.Categorical(dat['Category']).labels
crime_cat_column_=['Category']
cols=cols+crime_cat_column_
crime_cat=pd.get_dummies(dat['Category'])
crime_cat_columns=['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY',
                'BURGLARY', 'DISORDERLY CONDUCT',
                'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC',
                'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
                'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD',
                'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT',
                'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
                'NON-CRIMINAL', 'OTHER OFFENSES',
                'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION',
                'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
                'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
                'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
                'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS',
                'VANDALISM', 'VEHICLE THEFT', 'WARRANTS',
                'WEAPON LAWS']
cols=cols+crime_cat_columns
cols=cols+['X','Y']
district_cat=pd.get_dummies(dat['PdDistrict'])
cols=cols+list(district_cat.columns)
X_train=pd.DataFrame(np.column_stack((time_of_day,weekday_cat,crime_cat_,crime_cat,
    dat['X'],dat['Y'],district_cat)))
X_train.columns=cols

#remove objects
del dat
#################################################################################
#################################################################################
#visualization
#################################################################################
#################################################################################
%pylab
import matplotlib.pyplot as plt
import matplotlib as mpl

#################################################################################
#################################################################################
#scatter plot of crimes, with police districts color coded
x_val={}
y_val={}
for i in district_cat:
    x_val[i]=X_train[X_train[i]==1.0]['X']
    y_val[i]=X_train[X_train[i]==1.0]['Y']
n=len(x_val)
colors = mpl.cm.rainbow(np.linspace(0, 1, n))
fig, ax = plt.subplots()
for color, key_ in zip(colors,x_val.keys()):
    ax.scatter(x_val[key_], y_val[key_], color=color)
plt.show()

#################################################################################
#################################################################################
#scatter plot of crimes, with category color coded
x_val={}
y_val={}
for i in crime_cat_columns:
    x_val[i]=X_train[X_train[i]==1.0]['X']
    y_val[i]=X_train[X_train[i]==1.0]['Y']
n=len(x_val)
colors = mpl.cm.rainbow(np.linspace(0, 1, n))
fig, ax = plt.subplots()
for color, key_ in zip(colors,x_val.keys()):
    ax.scatter(x_val[key_], y_val[key_], color=color)
plt.show()

#################################################################################
#################################################################################
#histograms for each hour of day
for i in range(24):
    # X_train[X_train['Time']>=i*3600 & X_train['Time']<(i+1)*3600][crime_cat_column_].hist()
    l_time=i*3600
    u_time=(i+1)*3600
    X_train[((X_train['Time']>=l_time) & (X_train['Time']<u_time))][crime_cat_column_].hist(bins=39)
    plt.suptitle("Hour %i" %i)



#################################################################################
#################################################################################
#histograms for each hour of day
day_crime=np.zeros((24,39))
for i in range(24):
    # X_train[X_train['Time']>=i*3600 & X_train['Time']<(i+1)*3600][crime_cat_column_].hist()
    l_time=i*3600
    u_time=(i+1)*3600
    day_crime[i,:]=X_train[((X_train['Time']>=l_time) & (X_train['Time']<u_time))][crime_cat_columns].astype(float).sum(axis=0)
    #normalize by total number of crimes
    day_crime[i,:]/=day_crime[i,:].sum()
#dataframe of crime category proportions for each hour
day_crime=pd.DataFrame(data=day_crime,columns=crime_cat_columns)
#################################################################################
#################################################################################
x_min=X_train['X'].min()
x_max=X_train['X'].max()
y_min=X_train['Y'].min()
y_max=X_train['Y'].max()

m=20
n=20
x_ls=np.linspace(x_min,x_max,m)
y_ls=np.linspace(y_min,y_max,n)

crime_grid=np.zeros(((m-1)*(n-1),39+2))
ind=0
for i in range(m-1):
    for j in range(n-1):
        crime_grid[ind,0]=i
        crime_grid[ind,1]=j
        crime_grid[ind,2:]=X_train[((X_train['X']>=x_ls[i]) & (X_train['X']<x_ls[i+1]) 
            & (X_train['Y']>=y_ls[j]) & (X_train['Y']<y_ls[j+1]))][crime_cat_columns].sum(axis=0)
        ind+=1
crime_grid=pd.DataFrame(data=crime_grid,columns=['X','Y']+crime_cat_columns)