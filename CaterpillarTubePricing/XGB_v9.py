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
def xgb_train(ind_valid,ind_holdout,log_ind,train_X=None,train_Y=None,valid_X=None,valid_Y=None,holdout_X=None,holdout_Y=None,
    param={},num_round=None,y_pow=16.0):
    ####################################################################################
    if log_ind==0:
        xg_train = xgb.DMatrix( train_X, label=np.power(train_Y,1.0/y_pow))
    else:
        xg_train = xgb.DMatrix( train_X, label=np.log1p(train_Y))

    xg_test = xgb.DMatrix(test_X)
    # setup parameters for xgboost
    watchlist = [ (xg_train,'train')]
    bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
    n_tree = bst.best_iteration
    # print bst.get_fscore()

    # get prediction
    pred1 = bst.predict( xg_test,ntree_limit=n_tree );

    if log_ind==0:
        pred1 = np.power(pred1,y_pow)
    else:
        pred1= np.expm1(pred1)

    if ind_valid==1:
        xg_valid = xgb.DMatrix(valid_X)
        pred1_valid = bst.predict( xg_valid,ntree_limit=n_tree );

        if log_ind==0:
            pred1_valid= np.power(pred1_valid,y_pow)
        else:
            pred1_valid= np.expm1(pred1_valid)

        if ind_holdout==1:
            #holdout
            xg_holdout = xgb.DMatrix(holdout_X)
            pred1_holdout = bst.predict( xg_holdout,ntree_limit=n_tree );
            if log_ind==0:
                pred1_holdout= np.power(pred1_holdout,y_pow)
            else:
                pred1_holdout= np.expm1(pred1_holdout)
            return pred1_valid, pred1_holdout, pred1
        else:
            return pred1_valid, pred1
    else:
        return pred1
####################################################################################
####################################################################################
#train model on fraction of data set and validate on left out data
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
#enter 1 to use a holdout set
ind_holdout=1
n_folds=6
list_sets=[]
assembly_ids=dfx['tube_assembly_id'].unique()
if ind_valid==1:

    holdout_indices=[]
    if ind_holdout==1:
        #generate holdout set
        frac=0.10
        n_holdout=int(frac*sz[0])
        n_temp=0
        while n_temp<n_holdout:
            tube_id=np.random.choice(assembly_ids,size=1,replace=False)
            assembly_ids=list(set(assembly_ids)-set(tube_id))
            holdout_indices=holdout_indices+list(dfx[dfx['tube_assembly_id']==tube_id[0]].index)
            n_temp=len(holdout_indices)

    for i in range(n_folds):
        #generate validation set
        frac=0.15
        n_valid=int(frac*sz[0])

        n_temp=0
        valid_indices=[]
        while n_temp<n_valid:
            n_ids=len(assembly_ids)
            if n_ids>0:
                tube_id=np.random.choice(assembly_ids,size=1,replace=False)
                assembly_ids=list(set(assembly_ids)-set(tube_id))
                valid_indices=valid_indices+list(dfx[dfx['tube_assembly_id']==tube_id[0]].index)
                n_temp=len(valid_indices)
            else:
                print "assembly ids exhausted"
                n_temp=np.Inf

        #list of validation and holdout indices
        non_train_indices=valid_indices+holdout_indices
        #generate indices for training
        train_indices=list(set(dfx.index)-set(non_train_indices))
        list_sets.append((train_indices,valid_indices))

    if ind_holdout==1:
        list_sets.append(holdout_indices)
else:

    train_indices=range(dfx.shape[0])
    list_sets.append(train_indices)

####################################################################################
####################################################################################
####################################################################################



####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

dfx.drop('tube_assembly_id',axis=1, inplace=True)

# num_round_vec=[4500,3000,1500]
num_round_vec=[5000]

#model ranking (by rmse): 2,4,5,1,6,3,0
y_pow_vec=[16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0]
log_ind_vec=[0,1,0,0,0,0,1,1,1]
min_child_weight=[6,6,40,1,10,40,6,20,30]
subsample=[0.7,0.7,0.5,0.9,0.5,0.8,0.5,0.8,0.6]
colsample_bytree=[0.6,0.6,0.5,0.9,0.5,0.8,0.5,0.8,0.6]
max_depth=[8,8,5,30,12,5,15,8,6]

# models=[0,1,3,4,7,8]
models=[0,8]
n_models=len(models)
n_round=len(num_round_vec)


if ind_holdout==1:
    holdout_indices=list_sets[n_folds]
    holdout_X=np.array(dfx.ix[holdout_indices])
    holdout_Y=np.array(df_Y.ix[holdout_indices])

rmse_valid_mat=np.zeros((n_folds,n_round*n_models+1))
rmse_holdout_mat=np.zeros((n_folds,n_round*n_models+1))
rmse_valid_blend=np.zeros((n_folds,3))
rmse_holdout_blend=np.zeros((n_folds,3))
preds_test_mat_LR=np.zeros((test_X.shape[0],n_folds))
preds_test_mat_Ridge=np.zeros((test_X.shape[0],n_folds))
preds_holdout_mat_LR=np.zeros((holdout_X.shape[0],n_folds))
preds_holdout_mat_Ridge=np.zeros((holdout_X.shape[0],n_folds))


for i in range(n_folds):

    #save test set predictions in a matrix

    train_indices=list_sets[i][0]
    train_X=np.array(dfx.ix[train_indices])
    train_Y=np.array(df_Y.ix[train_indices])
    if ind_valid==1:
        valid_indices=list_sets[i][1]
        valid_X=np.array(dfx.ix[valid_indices])
        valid_Y=np.array(df_Y.ix[valid_indices])

        X_mat_test=np.zeros((test_X.shape[0],n_round*n_models))
        X_mat_valid=np.zeros((valid_X.shape[0],n_round*n_models))
        X_mat_holdout=np.zeros((holdout_X.shape[0],n_round*n_models))

        rmse_valid_mat[i,0]=i
        rmse_holdout_mat[i,0]=i

        iind=0
        for k in range(n_models):

            for kk in range(len(num_round_vec)):
                param = {}
                # # use softmax multi-class classification
                # param['objective'] = 'multi:softmax'
                # scale weight of positive examples

                param["objective"] = "reg:linear"
                param["eta"] = 0.02
                param["min_child_weight"] = min_child_weight[k]
                param["subsample"] = subsample[k]
                param["colsample_bytree"] = colsample_bytree[k]
                param["scale_pos_weight"] = 0.8
                # param['gamma'] = 5
                param["silent"] = 1
                param["max_depth"] = max_depth[k]
                param['nthread'] = 4
                param['num_class'] = 1
                param["max_delta_step"]=2 

                log_ind=log_ind_vec[k]
                num_round=int(num_round_vec[kk])
                y_pow=y_pow_vec[k]

                pred1_valid, pred1_holdout, pred1=xgb_train(ind_valid,ind_holdout,log_ind,train_X,train_Y,valid_X,valid_Y,holdout_X,holdout_Y,
                    param,num_round,y_pow)

                X_mat_test[:,iind]=np.log(pred1.ravel())
                X_mat_valid[:,iind]=np.log(pred1_valid.ravel())
                X_mat_holdout[:,iind]=np.log(pred1_holdout.ravel())

                rmse_valid_mat[i,iind+1]=rmse_log(valid_Y,pred1_valid)
                rmse_holdout_mat[i,iind+1]=rmse_log(holdout_Y,pred1_holdout)
                iind+=1
        ####################################################################################
        ####################################################################################
        alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
        RidgeModel = RidgeCV(alphas=alphas, normalize=True, cv=5)

        Ridge_fit=RidgeModel.fit(X_mat_valid,np.log(valid_Y))
        preds_test_Ridge=np.exp(Ridge_fit.predict(X_mat_test))
        preds_test_mat_Ridge[:,i]=preds_test_Ridge.ravel()
        preds_valid_Ridge=np.exp(Ridge_fit.predict(X_mat_valid))
        preds_holdout_Ridge=np.exp(Ridge_fit.predict(X_mat_holdout))
        preds_holdout_mat_Ridge[:,i]=preds_holdout_Ridge.ravel()

        rmse_valid_blend[i,0]=i
        rmse_valid_blend[i,1]=rmse_log(valid_Y,preds_valid_Ridge)
        rmse_holdout_blend[i,0]=i
        rmse_holdout_blend[i,1]=rmse_log(holdout_Y,preds_holdout_Ridge)
        ####################################################################################
        ####################################################################################
        LRmodel=LinearRegression(
            fit_intercept=False, 
            normalize=False, 
            copy_X=True)

        #fit blender using validation set
        LR_fit=LRmodel.fit(X_mat_valid,np.log(valid_Y))
        preds_test_LR=np.exp(LR_fit.predict(X_mat_test))
        preds_test_mat_LR[:,i]=preds_test_LR.ravel()
        preds_valid_LR=np.exp(LR_fit.predict(X_mat_valid))
        preds_holdout_LR=np.exp(LR_fit.predict(X_mat_holdout))
        preds_holdout_mat_LR[:,i]=preds_holdout_LR.ravel()

        rmse_valid_blend[i,2]=rmse_log(valid_Y,preds_valid_LR)
        rmse_holdout_blend[i,2]=rmse_log(holdout_Y,preds_holdout_LR)

####################################################################################
print rmse_valid_blend.mean(0)
print rmse_holdout_blend.mean(0)


LRmodel=LinearRegression(
    fit_intercept=False, 
    normalize=False, 
    copy_X=True)

#fit blender using validation set
LR_fit=LRmodel.fit(np.log(preds_holdout_LR),np.log(holdout_Y))
y_out=np.exp(LR_fit.predict(np.log(preds_holdout_LR)))
print rmse_log(holdout_Y,y_out)


alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

RidgeModel = RidgeCV(alphas=alphas, normalize=True, cv=5)

Ridge_fit=RidgeModel.fit(np.log(preds_holdout_mat_Ridge),np.log(holdout_Y))
y_out=np.exp(Ridge_fit.predict(np.log(preds_holdout_mat_Ridge)))
print rmse_log(holdout_Y,y_out)
# y_out=preds_test_mat_LR.mean(1)
# y_out=preds_test_mat.mean(1)

y_out=np.exp(Ridge_fit.predict(np.log(preds_test_mat_Ridge)))
###################################################################################
df=pd.DataFrame(y_out)
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)


####################################################################################
####################################################################################
#train model on full set
####################################################################################
####################################################################################
####################################################################################

test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))
indices=[i+1 for i in range(test_X.shape[0])]
dfx=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
df_Y=pd.io.parsers.read_table('Y_train.csv',sep=',',header=False)
sz = dfx.shape

dfx.drop('tube_assembly_id',axis=1, inplace=True)

train_X=np.array(dfx)
train_Y=np.array(df_Y)
#enter 1 to use a validation set
ind_valid=0
#enter 1 to use a holdout set
ind_holdout=0

X_mat_test=np.zeros((test_X.shape[0],n_round*n_models))

iind=0
for k in range(n_models):

    for kk in range(len(num_round_vec)):
        param = {}
        # # use softmax multi-class classification
        # param['objective'] = 'multi:softmax'
        # scale weight of positive examples

        param["objective"] = "reg:linear"
        param["eta"] = 0.02
        param["min_child_weight"] = min_child_weight[k]
        param["subsample"] = subsample[k]
        param["colsample_bytree"] = colsample_bytree[k]
        param["scale_pos_weight"] = 0.8
        # param['gamma'] = 5
        param["silent"] = 1
        param["max_depth"] = max_depth[k]
        param['nthread'] = 4
        param['num_class'] = 1
        param["max_delta_step"]=2 

        log_ind=log_ind_vec[k]
        num_round=int(num_round_vec[kk])
        y_pow=y_pow_vec[k]

        pred1=xgb_train(ind_valid,ind_holdout,log_ind,train_X,train_Y,valid_X=None,valid_Y=None,
            holdout_X=None,holdout_Y=None,param,num_round,y_pow)

        X_mat_test[:,iind]=np.log(pred1.ravel())
        iind+=1
####################################################################################
####################################################################################