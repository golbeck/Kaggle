#train on one full subset of the data (no separate trainers for bracket pricing)

import os
import sys
import time
import datetime
import operator
import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import xgboost as xgb

'''
for details on using xgboost, see:
https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
'''
####################################################################################
####################################################################################
####################################################################################
# pwd_temp=os.getcwd()
# # dir1='/home/sgolbeck/workspace/Kaggle/CaterpillarTubePricing'
# dir1='/home/golbeck/Workspace/Kaggle/CaterpillarTubePricing'
# dir1=dir1+'/data' 
# if pwd_temp!=dir1:
#     os.chdir(dir1)

X_dat=np.array(pd.io.parsers.read_table('X_train.csv',sep=',',header=False))
df_temp=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
col_names=df_temp.columns
del df_temp
test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))

indices=[i+1 for i in range(test_X.shape[0])]

n=X_dat.shape[0]
sz = X_dat.shape
####################################################################################
####################################################################################
####################################################################################
#train model on fraction of data set and validate on left out data
####################################################################################
####################################################################################
####################################################################################
sz = X_dat.shape
param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softmax'
# scale weight of positive examples

param["objective"] = "reg:linear"
param["eta"] = 0.01
param["min_child_weight"] = 10
param["subsample"] = 0.55
param["colsample_bytree"] = 0.87
param["scale_pos_weight"] = 1.0
# param['gamma'] = 5
param["silent"] = 1
param["max_depth"] = 30
param['nthread'] = 4
n_class=1
param['num_class'] = n_class
####################################################################################
####################################################################################
####################################################################################
# test algorithm out of sample
####################################################################################
####################################################################################
####################################################################################
Y_dat=np.array(pd.io.parsers.read_table('Y_train.csv',sep=',',header=False))
y_pow=16.0
num_round = 7000
Y_dat=np.power(Y_dat,1.0/y_pow)

frac=0.95
train_X = X_dat[:int(sz[0] * frac), :]
train_Y = Y_dat[:int(sz[0] * frac)]
valid_X = X_dat[int(sz[0] * frac):, :]
valid_Y = Y_dat[int(sz[0] * frac):]

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_valid,ntree_limit=n_tree );
pred1 = np.power(pred,y_pow)


####################################################################################
####################################################################################
####################################################################################
Y_dat=np.array(pd.io.parsers.read_table('Y_train.csv',sep=',',header=False))
num_round = 7000
Y_dat=np.log1p(Y_dat)

frac=0.95
train_X = X_dat[:int(sz[0] * frac), :]
train_Y = Y_dat[:int(sz[0] * frac)]
valid_X = X_dat[int(sz[0] * frac):, :]
valid_Y = Y_dat[int(sz[0] * frac):]

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_valid,ntree_limit=n_tree );
pred2 = np.expm1(pred)



preds=0.5*pred1+0.5*pred2
valid_Y = np.expm1(valid_Y)
print ('prediction error=%f' % np.sqrt(sum( (np.log1p(preds[i])-np.log1p(valid_Y[i]))**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))


####################################################################################
####################################################################################
####################################################################################
# train on full dataset
train_X = X_dat
Y_dat=np.array(pd.io.parsers.read_table('Y_train.csv',sep=',',header=False))
####################################################################################
num_round = 4000
y_pow=16.0
train_Y=np.power(Y_dat,1.0/y_pow)

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_test,ntree_limit=n_tree );
pred1 = np.power(pred,y_pow)

####################################################################################
num_round = 2000
train_Y=np.log1p(Y_dat)

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_test,ntree_limit=n_tree );
pred2 = pred
# pred2 = np.expm1(pred)

####################################################################################
num_round = 3000
train_Y=np.log1p(Y_dat)

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_test,ntree_limit=n_tree );
pred3 = pred
# pred2 = np.expm1(pred)
####################################################################################
num_round = 4000
train_Y=np.log1p(Y_dat)

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_test,ntree_limit=n_tree );
pred4 = pred
# pred2 = np.expm1(pred)


preds=0.4*(pred1)+0.1*np.expm1(pred2)+0.1*np.expm1(pred3)+0.4*np.expm1(pred4)
df=pd.DataFrame(preds)
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)
