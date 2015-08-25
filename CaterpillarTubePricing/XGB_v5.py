# Stacking
# 0) assume a train and test set
# 1) partition training data into K folds
# 2) select K-1 folds. Call this set A
# 3) call the LOF (left out fold) set B
# 3) for set A, fit layer 1 models on K-2 folds, leaving out one fold (A'), and generate predictions on this LOF
# 4) for all layer 1 models, stack/blend the models using predictions on the A' folds. 
# 5) fit all layer 1 models to the full set A, then blend using each of the fits on the A' folds. Generate
# predictions on set B. This gives an OOS performance measure.
# 6) fit all layer 1 models to A+B, apply the chosen blender, and generate predictions on the test set

# Note: sets 2-5 can be repeated with a different set B, generating different blenders

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
####################################################################################
####################################################################################
####################################################################################
def rmse_log(Y_true,Y_pred):
    temp=np.sqrt(sum( (np.log1p(Y_pred[m]) - np.log1p(Y_true[m]))**2 for m in range(len(Y_true))) / float(len(Y_true)))
    return temp
####################################################################################
####################################################################################
####################################################################################
def xgb_train_mod(y_pow,train_X,Y_dat,valid_X,holdout_X,param,num_round,log_ind):
    #transform training labels
    if log_ind==1.0:
        train_Y = np.log1p(Y_dat)
    else:
        train_Y = np.power(Y_dat,1.0/y_pow)

    #setup train, validation and holdout inputs to XGB
    xg_train = xgb.DMatrix( train_X, label=train_Y)
    xg_valid = xgb.DMatrix(valid_X)
    xg_holdout = xgb.DMatrix(holdout_X)

    #train model
    watchlist = [ (xg_train,'train') ]
    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50 );
    n_tree = bst.best_iteration

    #predict on validation set, which will be used for fitting the blended model
    pred_valid = bst.predict( xg_valid, ntree_limit=n_tree )

    #transform predictions
    if log_ind==1.0:
        pred_valid = np.expm1(pred_valid)
    else:
        pred_valid = np.power(pred_valid,y_pow)

    #predict on holdout set
    pred_holdout = bst.predict( xg_holdout, ntree_limit=n_tree )

    #transform predictions
    if log_ind==1.0:
        pred_holdout = np.expm1(pred_holdout)
    else:
        pred_holdout = np.power(pred_holdout,y_pow)

    #return predictions
    return [pred_valid,pred_holdout]
####################################################################################
####################################################################################
####################################################################################
def split_by_id(n_holdout,dfx,dfy,assembly_ids):
    #n_holdout: size of holdout set; function will return a set near this size
    #df1: dataframe that will be split
    #assembly_ids: split will be generated using this list of unique ids

    #dataframe for 
    holdout_indices=[]
    n_temp=0
    while n_temp<n_holdout:
        tube_id=np.random.choice(assembly_ids,size=1,replace=False)
        assembly_ids=list(set(assembly_ids)-set(tube_id))
        holdout_indices=holdout_indices+list(dfx[dfx['tube_assembly_id']==tube_id[0]].index)
        n_temp=len(holdout_indices)

    #generate indices of set used for training
    train_indices=list(set(dfx.index)-set(holdout_indices))
    #generate numpy arrays for training features and labels
    X_dat=dfx.ix[train_indices]
    Y_dat=dfy.ix[train_indices]
    holdout_X=dfx.ix[holdout_indices]
    holdout_Y=dfy.ix[holdout_indices]
    return [X_dat,Y_dat,holdout_X,holdout_Y]



####################################################################################
####################################################################################
####################################################################################

param = {}
param["objective"] = "reg:linear"
param["eta"] = 0.01
param["min_child_weight"] = 10
param["subsample"] = 0.55
param["colsample_bytree"] = 0.87
param["scale_pos_weight"] = 1.0
param["silent"] = 1
param["max_depth"] = 30
param['nthread'] = 4
param['num_class'] = 1
num_round=7000



df1=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
df_Y=pd.io.parsers.read_table('Y_train.csv',sep=',',header=False)

test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))

indices=[i+1 for i in range(test_X.shape[0])]
####################################################################################
####################################################################################

K_folds=5
frac=1.0/K_folds
sz=df1.shape
n_holdout=int(frac*sz[0])

#list of lists of models
models=[[2.0,0.0],[3.0,0.0],[16.0,0.0],[16.0,1.0]]
n_models=len(models)
blender_rmse_enet=[]
blender_rmse_LR=[]
blender_fits_enet=[]
blender_fits_LR=[]
rmse_matrix=np.zeros((n_models*K_folds*(K_folds-2),2))
#loop over holdout (each fold used as holdout set)
iind=0
temp=[]
for i in range(K_folds):
    #generate training set of size (K_folds-1) and holdout set of size 1 fold
    assembly_ids=df1['tube_assembly_id'].unique()
    X_dat,Y_dat,holdout_X,holdout_Y=split_by_id(n_holdout,df1,df_Y,assembly_ids)
    holdout_X.drop('tube_assembly_id',axis=1, inplace=True)
    holdout_X=np.array(holdout_X)
    holdout_Y=np.array(holdout_Y)
    assembly_ids=X_dat['tube_assembly_id'].unique()
    #for each 
    for j in range(K_folds-2):
        #generate training and validation set
        train_X,train_Y,valid_X,valid_Y=split_by_id(n_holdout,X_dat,Y_dat,assembly_ids)
        #drop splitting criterion
        train_X.drop('tube_assembly_id',axis=1, inplace=True)
        valid_X.drop('tube_assembly_id',axis=1, inplace=True)
        train_X=np.array(train_X)
        train_Y=np.array(train_Y)
        valid_X=np.array(valid_X)
        valid_Y=np.array(valid_Y)


        X_net_valid=np.zeros((valid_X.shape[0],n_models))
        X_net_holdout=np.zeros((holdout_X.shape[0],n_models))
        #loop over layer 1 models
        k=0
        for y_pow,log_ind in models:
            #predict on given validation set
            pred_valid,pred_holdout=xgb_train_mod(y_pow,train_X,train_Y,valid_X,holdout_X,param,num_round,log_ind)
            X_net_valid[:,k]=pred_valid
            rmse_matrix[iind,0]=rmse_log(valid_Y,pred_valid)
            X_net_holdout[:,k]=pred_holdout
            rmse_matrix[iind,1]=rmse_log(holdout_Y,pred_holdout)
            k+=1
            iind+=1

        enet=ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=False, normalize=False, 
            precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, 
            positive=False)
        LRmodel=LinearRegression(
            fit_intercept=False, 
            normalize=False, 
            copy_X=True)
        #fit blender using validation set
        enet_mod=enet.fit(X_net_valid,valid_Y)
        #save model along with holdout set performance measure
        blender_fits_enet.append(enet_mod)
        pred_holdout=enet_mod.predict(X_net_holdout)
        #generate OOS performance measure on holdout set
        blender_rmse_enet.append([enet_mod,rmse_log(holdout_Y,pred_holdout)])


        #fit blender using validation set
        LR_mod=LRmodel.fit(X_net_valid,valid_Y)
        #save model along with holdout set performance measure
        blender_fits_LR.append(LR_mod)
        pred_holdout=LR_mod.predict(X_net_holdout)
        #generate OOS performance measure on holdout set
        blender_rmse_LR.append([LR_mod,rmse_log(holdout_Y,pred_holdout)])

        temp.append(X_net_holdout)
    print blender_rmse_enet
    print [a.coef_ for a in blender_fits_enet]
    print [a.coef_ for a in blender_fits_LR]
    print "iteration %i out of %g" % (i+1, K_folds)


####################################################################################
####################################################################################

K_folds=5
frac=1.0/K_folds
sz=df1.shape
n_holdout=int(frac*sz[0])
blender_rmse_full_enet=[]
blender_rmse_full_LR=[]
#choose different holdout sets
for i in range(K_folds):
    
    #generate training set of size (K_folds-1) and holdout set of size 1 fold
    assembly_ids=df1['tube_assembly_id'].unique()
    train_X,train_Y,holdout_X,holdout_Y=split_by_id(n_holdout,df1,df_Y,assembly_ids)
    holdout_X.drop('tube_assembly_id',axis=1, inplace=True)
    holdout_X=np.array(holdout_X)
    holdout_Y=np.array(holdout_Y)    
    train_X.drop('tube_assembly_id',axis=1, inplace=True)
    train_X=np.array(train_X)
    train_Y=np.array(train_Y)
    #loop over layer 1 models
    k=0
    X_net_holdout=np.zeros((holdout_X.shape[0],n_models))
    for y_pow,log_ind in models:
        #predict on given validation set
        pred_valid,pred_holdout=xgb_train_mod(y_pow,train_X,train_Y,holdout_X,holdout_X,param,num_round,log_ind)
        X_net_holdout[:,k]=pred_holdout
        k+=1

    for j in range(len(blender_fits_enet)):
        mod=blender_fits_enet[j]
        pred_holdout=mod.predict(X_net_holdout)
        blender_rmse_full_enet.append([mod,rmse_log(holdout_Y,pred_holdout)])

    for j in range(len(blender_fits_LR)):
        mod=blender_fits_LR[j]
        pred_holdout=mod.predict(X_net_holdout)
        blender_rmse_full_LR.append([mod,rmse_log(holdout_Y,pred_holdout)])


####################################################################################
####################################################################################
####################################################################################

param = {}
param["objective"] = "reg:linear"
param["eta"] = 0.01
param["min_child_weight"] = 10
param["subsample"] = 0.55
param["colsample_bytree"] = 0.87
param["scale_pos_weight"] = 1.0
param["silent"] = 1
param["max_depth"] = 30
param['nthread'] = 4
param['num_class'] = 1
num_round=2000



df1=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
df_Y=pd.io.parsers.read_table('Y_train.csv',sep=',',header=False)

test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))

indices=[i+1 for i in range(test_X.shape[0])]
####################################################################################
####################################################################################

models=[[16.0,0.0],[16.0,1.0]]
n_models=len(models)

K_folds=5
frac=1.0/K_folds
sz=df1.shape
n_holdout=int(frac*sz[0])
blender_rmse_full_enet=[]
blender_rmse_full_LR=[]
blender_rmse_full_ridge=[]
y_test_enet=np.zeros((test_X.shape[0],K_folds))
y_test_LR=np.zeros((test_X.shape[0],K_folds))
y_test_ridge=np.zeros((test_X.shape[0],K_folds))
rmse_matrix=np.zeros((K_folds*n_models,3))
#choose different holdout sets
for i in range(K_folds):
    
    #generate training set of size (K_folds-1) and holdout set of size 1 fold
    assembly_ids=df1['tube_assembly_id'].unique()
    train_X,train_Y,holdout_X,holdout_Y=split_by_id(n_holdout,df1,df_Y,assembly_ids)
    holdout_X.drop('tube_assembly_id',axis=1, inplace=True)
    holdout_X=np.array(holdout_X)
    holdout_Y=np.array(holdout_Y)    
    train_X.drop('tube_assembly_id',axis=1, inplace=True)
    train_X=np.array(train_X)
    train_Y=np.array(train_Y)
    #loop over layer 1 models
    k=0
    X_net_holdout=np.zeros((holdout_X.shape[0],n_models))
    X_net_test=np.zeros((test_X.shape[0],n_models))
    for y_pow,log_ind in models:
        #predict on given validation set
        pred_holdout,pred_test=xgb_train_mod(y_pow,train_X,train_Y,holdout_X,test_X,param,num_round,log_ind)
        X_net_holdout[:,k]=pred_holdout
        X_net_test[:,k]=pred_test
        rmse_matrix[i*n_models+k,0]=y_pow
        rmse_matrix[i*n_models+k,1]=log_ind
        rmse_matrix[i*n_models+k,2]=rmse_log(holdout_Y,pred_holdout)
        k+=1

    
    enet=ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=False, normalize=False, 
        precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, 
        positive=False)
    LRmodel=LinearRegression(
        fit_intercept=False, 
        normalize=False, 
        copy_X=True)

    Ridge = RidgeCV(alphas=alphas, normalize=True, cv=5)

    ####################################################################################
    #fit blender using validation set
    enet_mod=enet.fit(X_net_holdout,holdout_Y)
    #save model along with holdout set performance measure
    pred_holdout=enet_mod.predict(X_net_holdout)
    #generate OOS performance measure on holdout set
    blender_rmse_enet.append([enet_mod,rmse_log(holdout_Y,pred_holdout)])
    #test set predictions
    y_test_enet[:,i]=enet_mod.predict(X_net_test)

    ####################################################################################
    #fit blender using validation set
    LR_mod=LRmodel.fit(X_net_holdout,holdout_Y)
    #save model along with holdout set performance measure
    pred_holdout=LR_mod.predict(X_net_holdout)
    #generate OOS performance measure on holdout set
    blender_rmse_LR.append([LR_mod,rmse_log(holdout_Y,pred_holdout)])
    #test set predictions
    y_test_LR[:,i]=LR_mod.predict(X_net_test).reshape(test_X.shape[0])

    ####################################################################################
    alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    ridge_fit=Ridge.fit(X_net_holdout, holdout_Y)    
    pred_holdout=ridge_fit.predict(X_net_holdout)
    #generate OOS performance measure on holdout set
    blender_rmse_full_ridge.append([ridge_fit,rmse_log(holdout_Y,pred_holdout)])
    y_test_ridge[:,i] = ridge_fit.predict(X_net_test).reshape(test_X.shape[0])


y_test=np.column_stack((y_test_enet,y_test_LR,y_test_ridge))


df=pd.DataFrame(y_test.mean(1))
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)