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
param['num_class'] = 1
num_round = 1000

####################################################################################
####################################################################################
####################################################################################

test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))
indices=[i+1 for i in range(test_X.shape[0])]
dfx=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
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
    train_X=np.array(dfx.ix[train_indices])
    train_Y=np.array(df_Y.ix[train_indices])
    valid_X=np.array(dfx.ix[valid_indices])
    valid_Y=np.array(df_Y.ix[valid_indices])
    holdout_X=np.array(dfx.ix[holdout_indices])
    holdout_Y=np.array(df_Y.ix[holdout_indices])
else:

    dfx.drop('tube_assembly_id',axis=1, inplace=True)
    train_X=np.array(dfx)
    train_Y=np.array(df_Y)

summary_valid=[]
summary_holdout=[]
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
pred = bst.predict( xg_test,ntree_limit=n_tree );
pred1 = np.power(pred,y_pow)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X)
    pred_pow_valid = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_pow_valid= np.power(pred_pow_valid,y_pow)
    summary_valid.append([y_pow,rmse_log(valid_Y,pred_pow_valid)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X)
    pred_pow_holdout = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_pow_holdout= np.power(pred_pow_holdout,y_pow)
    summary_holdout.append([y_pow,rmse_log(holdout_Y,pred_pow_holdout)])
####################################################################################
xg_train = xgb.DMatrix( train_X, label=np.log1p(train_Y))
xg_test = xgb.DMatrix(test_X)
# setup parameters for xgboost
watchlist = [ (xg_train,'train')]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()

# get prediction
pred = bst.predict( xg_test,ntree_limit=n_tree );
pred2 = pred
pred2 = np.expm1(pred)

if ind_valid==1:
    xg_valid = xgb.DMatrix(valid_X)
    pred_log_valid = bst.predict( xg_valid,ntree_limit=n_tree );
    pred_log_valid= np.expm1(pred_log_valid)
    summary_valid.append(['log1p',rmse_log(valid_Y,pred_log_valid)])
    #holdout
    xg_holdout = xgb.DMatrix(holdout_X)
    pred_log_holdout = bst.predict( xg_holdout,ntree_limit=n_tree );
    pred_log_holdout= np.power(pred_log_holdout,y_pow)
    summary_holdout.append([y_pow,rmse_log(holdout_Y,pred_log_holdout)])


####################################################################################
num_tree_RF=1000
print "Training Random Forest model"
#RF classifier
RF = RandomForestRegressor(
        n_estimators=num_tree_RF,        #number of trees to generate
        n_jobs=4,               #run in parallel on all cores
        criterion="mse"
        ) 

RFmodel = RF.fit(train_X, train_Y.ravel())
pred_RF_test=RFmodel.predict(test_X)


if ind_valid==1:
    pred_RF_valid=RFmodel.predict(valid_X)
    summary_valid.append(['RF',rmse_log(valid_Y,pred_RF_valid)])
    pred_RF_holdout=RFmodel.predict(holdout_X)
    summary_holdout.append(['RF',rmse_log(holdout_Y,pred_RF_holdout)])
    
####################################################################################
num_tree_ET=500
print "Training Extremely Randomized Trees model"
seed=1234
ET=ExtraTreesRegressor(n_estimators=num_tree_ET, criterion='mse', max_depth=30, min_samples_split=10, 
        max_features=0.80, n_jobs=4, random_state=seed, verbose=0)
ETmodel=ET.fit(train_X,train_Y.ravel())
pred_ET_test=ETmodel.predict(test_X)

if ind_valid==1:
    pred_ET_valid=ETmodel.predict(valid_X)
    summary_valid.append(['ET',rmse_log(valid_Y,pred_ET_valid)])
    pred_ET_holdout=RFmodel.predict(holdout_X)
    summary_holdout.append(['ET',rmse_log(holdout_Y,pred_ET_holdout)])


if ind_valid==1:
    print summary_valid

####################################################################################
#select coefficients for layer 1 model by hand
coef=[0.60,0.25,0.10,0.05]
preds_valid=coef[0]*pred_pow_valid+coef[1]*pred_log_valid+coef[2]*pred_RF_valid+coef[3]*pred_ET_valid
print rmse_log(valid_Y,preds_valid)

preds=coef[0]*(pred1)+coef[1]*pred2+coef[2]*pred_RF_test+coef[3]*pred_ET_test


####################################################################################
#fit coefficients using linear regression on log validation set predictions
X_mat_valid=np.column_stack([np.log(pred_pow_valid),np.log(pred_log_valid),np.log(pred_RF_valid),np.log(pred_ET_valid)])
X_mat_holdout=np.column_stack([np.log(pred_pow_holdout),np.log(pred_log_holdout),np.log(pred_RF_holdout),np.log(pred_ET_holdout)])

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
X_mat_test=np.column_stack([np.log(pred1),np.log(pred2),np.log(pred_RF_test),np.log(pred_ET_test)])
preds_test=np.exp(LR_fit.predict(X_mat_test))


####################################################################################
df=pd.DataFrame(preds_test)
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)



