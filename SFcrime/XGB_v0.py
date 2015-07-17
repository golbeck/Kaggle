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

####################################################################################
####################################################################################
####################################################################################
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/SFcrime'
# dir1='/home/golbeck/Workspace/Kaggle/SFcrime'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

X_dat=np.loadtxt("X_train.gz",delimiter=",")
Y_dat=np.loadtxt("Y_train.gz",delimiter=",")
n=X_dat.shape[0]

#shuffle train set
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_dat)
np.random.set_state(rng_state)
np.random.shuffle(Y_dat)        

test_X=np.loadtxt("X_test.gz",delimiter=",")
####################################################################################
####################################################################################
####################################################################################
sz = X_dat.shape

frac=0.99
train_X = X_dat[:int(sz[0] * frac), :]
train_Y = Y_dat[:int(sz[0] * frac)]
valid_X = X_dat[int(sz[0] * frac):, :]
valid_Y = Y_dat[int(sz[0] * frac):]

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 20
param['silent'] = 1
param['nthread'] = 4
n_class=39
param['num_class'] = n_class

watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
num_round = 20
# bst = xgb.train(param, xg_train, num_round, watchlist );
# # get prediction
# pred = bst.predict( xg_valid );

# print ('predicting, classification error=%f' % (sum( int(pred[i]) != valid_Y[i] for i in range(len(valid_Y))) / float(len(valid_Y)) ))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
# yprob = bst.predict( xg_valid ).reshape( valid_X.shape[0], n_class )
# ylabel = np.argmax(yprob, axis=1)

# print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))


yprob = bst.predict( xg_test ).reshape( test_X.shape[0], n_class )

####################################################################################
####################################################################################
####################################################################################
#output test set probabilities to csv file
columns=['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY',
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
df = pd.DataFrame(columns=['Id']+columns)
df=pd.DataFrame(yprob,columns=columns)
df.insert(loc=0,column='Id',value=range(len(df)))
# np.savetxt("RF_predictions.csv.gz", df, delimiter=",")
df.to_csv("XGB_predictions.csv",sep=",",index=False)



####################################################################################
####################################################################################
####################################################################################
# save model
bst.save_model('xgb.model')
# load model and data in
# bst2 = xgb.Booster(model_file='xgb.model')
# dtest2 = xgb.DMatrix('dtest.buffer')
# preds2 = bst2.predict(dtest2)