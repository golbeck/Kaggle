#ensembling model

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

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

'''
for details on using xgboost, see:
https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
'''

####################################################################################
####################################################################################
####################################################################################
def rmse_log(Y_true,Y_pred):
    temp=np.sqrt(sum( (np.log1p(Y_pred[m]) - np.log1p(Y_true[m]))**2 for m in range(len(Y_true))) / float(len(Y_true)))
    return temp
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
#train model on fraction of data set and validate on left out data
####################################################################################
####################################################################################
####################################################################################

####################################################################################
####################################################################################
####################################################################################

test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))
test_X1=np.array(pd.io.parsers.read_table('X_test_143.csv',sep=',',header=False))
indices=[i+1 for i in range(test_X.shape[0])]
dfx=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
dfx1=pd.io.parsers.read_table('X_train_143.csv',sep=',',header=False)
df_Y=pd.io.parsers.read_table('Y_train.csv',sep=',',header=False)
sz = dfx.shape

#enter 1 to use a validation set
ind_valid=1

if ind_valid==1:
    #generate validation set
    frac=0.05
    n_valid=int(frac*sz[0])
    assembly_ids=dfx['tube_assembly_id'].unique()
    temp=pd.DataFrame()

    n_temp=0
    valid_indices=[]
    while n_temp<n_valid:
        tube_id=np.random.choice(assembly_ids,size=1,replace=False)
        assembly_ids=list(set(assembly_ids)-set(tube_id))
        valid_indices=valid_indices+list(dfx[dfx['tube_assembly_id']==tube_id[0]].index)
        n_temp=len(valid_indices)

    #generate holdout set
    frac=0.05
    n_holdout=int(frac*sz[0])
    n_temp=0
    holdout_indices=[]
    while n_temp<n_holdout:
        tube_id=np.random.choice(assembly_ids,size=1,replace=False)
        assembly_ids=list(set(assembly_ids)-set(tube_id))
        holdout_indices=holdout_indices+list(dfx[dfx['tube_assembly_id']==tube_id[0]].index)
        n_temp=len(holdout_indices)

    #list of validation and holdout indices
    non_train_indices=valid_indices+holdout_indices
    #generate indices for training
    train_indices=list(set(dfx.index)-set(non_train_indices))

    dfx.drop('tube_assembly_id',axis=1, inplace=True)
    dfx1.drop('tube_assembly_id',axis=1, inplace=True)
    train_X=np.array(dfx.ix[train_indices])
    train_X1=np.array(dfx1.ix[train_indices])
    train_Y=np.array(df_Y.ix[train_indices])
    valid_X=np.array(dfx.ix[valid_indices])
    valid_X1=np.array(dfx1.ix[valid_indices])
    valid_Y=np.array(df_Y.ix[valid_indices])
    holdout_X=np.array(dfx.ix[holdout_indices])
    holdout_X1=np.array(dfx1.ix[holdout_indices])
    holdout_Y=np.array(df_Y.ix[holdout_indices])
else:

    dfx.drop('tube_assembly_id',axis=1, inplace=True)
    dfx1.drop('tube_assembly_id',axis=1, inplace=True)
    train_X=np.array(dfx)
    train_X1=np.array(dfx1)
    train_Y=np.array(df_Y)

summary_valid=[]
summary_holdout=[]


####################################################################################
####################################################################################
####################################################################################
param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softmax'
# scale weight of positive examples

param["objective"] = "reg:linear"
param["eta"] = 0.005
param["min_child_weight"] = 10
param["subsample"] = 0.55
param["colsample_bytree"] = 0.87
param["scale_pos_weight"] = 1.0
# param['gamma'] = 5
param["silent"] = 1
param["max_depth"] = 30
param['nthread'] = 4
param['num_class'] = 1
# [[[16.0, array([ 0.22542065])],
#   ['log1p', array([ 0.23168171])],
#   [16.0, array([ 0.22532309])],
#   ['log1p', array([ 0.2315382])]],
#  [[16.0, array([ 0.21892988])],
#   ['log1p', array([ 0.23178437])],
#   [16.0, array([ 0.21892988])],
#   ['log1p', array([ 0.23178437])]]]


param = {}
param["objective"] = "reg:linear"
param["eta"] = 0.02
param["min_child_weight"] = 6
param["subsample"] = 0.7
param["colsample_bytree"] = 0.6
param["scale_pos_weight"] = 0.8
param["silent"] = 1
param["max_depth"] = 8
param["max_delta_step"]=2
param['nthread'] = 4
param['num_class'] = 1
####################################################################################
####################################################################################
####################################################################################
num_round = 2000
####################################################################################
####################################################################################
####################################################################################
y_pow=16.0

xg_train = xgb.DMatrix( train_X, label=np.power(train_Y,1.0/y_pow))
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred1 = bst.predict( xg_test,ntree_limit=n_tree );
pred1 = np.power(pred1,y_pow)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X)
    pred_pow1_valid = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_pow1_valid= np.power(pred_pow1_valid,y_pow)
    summary_valid.append([y_pow,rmse_log(valid_Y,pred_pow1_valid)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X)
    pred_pow1_holdout = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_pow1_holdout= np.power(pred_pow1_holdout,y_pow)
    summary_holdout.append([y_pow,rmse_log(holdout_Y,pred_pow1_holdout)])
####################################################################################
xg_train = xgb.DMatrix( train_X, label=np.log1p(train_Y))
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred2 = bst.predict( xg_test,ntree_limit=n_tree );
pred2 = np.expm1(pred2)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X)
    pred_log2_valid = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_log2_valid= np.expm1(pred_log2_valid)
    summary_valid.append(['log1p',rmse_log(valid_Y,pred_log2_valid)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X)
    pred_log2_holdout = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_log2_holdout= np.expm1(pred_log2_holdout)
    summary_holdout.append(['log1p',rmse_log(holdout_Y,pred_log2_holdout)])


####################################################################################
####################################################################################
####################################################################################
num_round = 5000
####################################################################################
####################################################################################
####################################################################################
y_pow=16.0

xg_train = xgb.DMatrix( train_X, label=np.power(train_Y,1.0/y_pow))
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred3 = bst.predict( xg_test,ntree_limit=n_tree );
pred3 = np.power(pred3,y_pow)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X)
    pred_pow3_valid = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_pow3_valid= np.power(pred_pow3_valid,y_pow)
    summary_valid.append([y_pow,rmse_log(valid_Y,pred_pow3_valid)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X)
    pred_pow3_holdout = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_pow3_holdout= np.power(pred_pow3_holdout,y_pow)
    summary_holdout.append([y_pow,rmse_log(holdout_Y,pred_pow3_holdout)])
####################################################################################
xg_train = xgb.DMatrix( train_X, label=np.log1p(train_Y))
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred4 = bst.predict( xg_test,ntree_limit=n_tree );
pred4 = np.expm1(pred4)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X)
    pred_log4_valid = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_log4_valid= np.expm1(pred_log4_valid)
    summary_valid.append(['log1p',rmse_log(valid_Y,pred_log4_valid)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X)
    pred_log4_holdout = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_log4_holdout= np.expm1(pred_log4_holdout)
    summary_holdout.append(['log1p',rmse_log(holdout_Y,pred_log4_holdout)])


####################################################################################
####################################################################################
####################################################################################
num_round = 2000
####################################################################################
####################################################################################
####################################################################################
y_pow=16.0

xg_train = xgb.DMatrix( train_X1, label=np.power(train_Y,1.0/y_pow))
xg_test = xgb.DMatrix(test_X1)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred1_143 = bst.predict( xg_test,ntree_limit=n_tree );
pred1_143 = np.power(pred1_143,y_pow)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X1)
    pred_pow1_valid_143 = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_pow1_valid_143= np.power(pred_pow1_valid_143,y_pow)
    summary_valid.append([y_pow,rmse_log(valid_Y,pred_pow1_valid_143)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X1)
    pred_pow1_holdout_143 = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_pow1_holdout_143= np.power(pred_pow1_holdout_143,y_pow)
    summary_holdout.append([y_pow,rmse_log(holdout_Y,pred_pow1_holdout_143)])
####################################################################################
xg_train = xgb.DMatrix( train_X1, label=np.log1p(train_Y))
xg_test = xgb.DMatrix(test_X1)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred2_143 = bst.predict( xg_test,ntree_limit=n_tree );
pred2_143 = np.expm1(pred2_143)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X1)
    pred_log2_valid_143 = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_log2_valid_143= np.expm1(pred_log2_valid_143)
    summary_valid.append(['log1p',rmse_log(valid_Y,pred_log2_valid_143)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X1)
    pred_log2_holdout_143 = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_log2_holdout_143= np.expm1(pred_log2_holdout_143)
    summary_holdout.append(['log1p',rmse_log(holdout_Y,pred_log2_holdout_143)])


####################################################################################
####################################################################################
####################################################################################
num_round = 5000
####################################################################################
####################################################################################
####################################################################################
y_pow=16.0

xg_train = xgb.DMatrix( train_X1, label=np.power(train_Y,1.0/y_pow))
xg_test = xgb.DMatrix(test_X1)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred3_143 = bst.predict( xg_test,ntree_limit=n_tree );
pred3_143 = np.power(pred3_143,y_pow)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X1)
    pred_pow3_valid_143 = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_pow3_valid_143= np.power(pred_pow3_valid_143,y_pow)
    summary_valid.append([y_pow,rmse_log(valid_Y,pred_pow3_valid_143)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X1)
    pred_pow3_holdout_143 = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_pow3_holdout_143= np.power(pred_pow3_holdout_143,y_pow)
    summary_holdout.append([y_pow,rmse_log(holdout_Y,pred_pow3_holdout_143)])
####################################################################################
xg_train = xgb.DMatrix( train_X1, label=np.log1p(train_Y))
xg_test = xgb.DMatrix(test_X1)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred4_143 = bst.predict( xg_test,ntree_limit=n_tree );
pred4_143 = np.expm1(pred4_143)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X1)
    pred_log4_valid_143 = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_log4_valid_143= np.expm1(pred_log4_valid_143)
    summary_valid.append(['log1p',rmse_log(valid_Y,pred_log4_valid_143)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X1)
    pred_log4_holdout_143 = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_log4_holdout_143= np.expm1(pred_log4_holdout_143)
    summary_holdout.append(['log1p',rmse_log(holdout_Y,pred_log4_holdout_143)])

####################################################################################
#fit coefficients using linear regression on log validation set predictions
X_mat_valid=np.column_stack([np.log(pred_pow1_valid),np.log(pred_log2_valid),np.log(pred_pow3_valid),np.log(pred_log4_valid),
    np.log(pred_pow1_valid_143),np.log(pred_log2_valid_143),np.log(pred_pow3_valid_143),np.log(pred_log4_valid_143)])
X_mat_holdout=np.column_stack([np.log(pred_pow1_holdout),np.log(pred_log2_holdout),np.log(pred_pow3_holdout),np.log(pred_log4_holdout),
    np.log(pred_pow1_holdout_143),np.log(pred_log2_holdout_143),np.log(pred_pow3_holdout_143),np.log(pred_log4_holdout_143)])

LRmodel=LinearRegression(
    fit_intercept=False, 
    normalize=False, 
    copy_X=True)


LR_fit=LRmodel.fit(X_mat_valid,np.log(valid_Y).ravel())
preds_valid=np.exp(LR_fit.predict(X_mat_valid))
print rmse_log(valid_Y,preds_valid)

preds_holdout=np.exp(LR_fit.predict(X_mat_holdout))
print rmse_log(holdout_Y,preds_holdout)

#apply to test set predictions
X_mat_test=np.column_stack([np.log(pred1),np.log(pred2),np.log(pred3),np.log(pred4),
    np.log(pred1_143),np.log(pred2_143),np.log(pred3_143),np.log(pred4_143)])
preds_test=np.exp(LR_fit.predict(X_mat_test))

c=0.80*np.array([0.38,0.09,0.42,0.11])
c1=0.20*np.array([0.38,0.09,0.42,0.11])
preds_test=c[0]*pred1+c[1]*pred2+c[2]*pred3+c[3]*pred4+c1[0]*pred1_143+c1[1]*pred2_143+c1[2]*pred3_143+c1[3]*pred4_143
preds_valid=c[0]*pred_pow1_valid+c[1]*pred_log2_valid+c[2]*pred_pow3_valid+c[3]*pred_log4_valid+c1[0]*pred_pow1_valid_143+c1[1]*pred_log2_valid_143+c1[2]*pred_pow3_valid_143+c1[3]*pred_log4_valid_143
# print summary_valid
print rmse_log(valid_Y,preds_valid)
preds_holdout=c[0]*pred_pow1_holdout+c[1]*pred_log2_holdout+c[2]*pred_pow3_holdout+c[3]*pred_log4_holdout+c1[0]*pred_pow1_holdout_143+c1[1]*pred_log2_holdout_143+c1[2]*pred_pow3_holdout_143+c1[3]*pred_log4_holdout_143
# print summary_holdout
print rmse_log(holdout_Y,preds_holdout)


preds_test=np.exp(X_mat_test_full)

c=0.80*np.array([0.38,0.09,0.42,0.11])
c1=0.20*np.array([0.38,0.09,0.42,0.11])
c2=np.array(list(c)+list(c1))
y_out=np.dot(preds_test,c2)

# preds_test=c[0]*pred1+c[1]*pred2+c[2]*pred3+c[3]*pred4+c1[0]*pred1_143+c1[1]*pred2_143+c1[2]*pred3_143+c1[3]*pred4_143
# ####################################################################################
df=pd.DataFrame(y_out)
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)


# X_mat_valid=np.column_stack([np.log1p(pred_pow1_valid),np.log1p(pred_log2_valid),np.log1p(pred_pow3_valid),np.log1p(pred_log4_valid)])
# X_mat_holdout=np.column_stack([np.log1p(pred_pow1_holdout),np.log1p(pred_log2_holdout),np.log1p(pred_pow3_holdout),np.log1p(pred_log4_holdout)])

# LRmodel1=LinearRegression(
#     fit_intercept=False, 
#     normalize=False, 
#     copy_X=True)


# LR_fit1=LRmodel1.fit(X_mat_valid,np.log1p(valid_Y).ravel())
# preds_valid=np.expm1(LR_fit1.predict(X_mat_valid))
# print rmse_log(valid_Y,preds_valid)

# preds_holdout=np.expm1(LR_fit1.predict(X_mat_holdout))
# print rmse_log(holdout_Y,preds_holdout)


# #apply to test set predictions
# X_mat_test=np.column_stack([np.log1p(pred1),np.log1p(pred2),np.log1p(pred3),np.log1p(pred4)])
# preds_test=np.expm1(LR_fit.predict(X_mat_test))