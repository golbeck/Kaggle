
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
from sklearn.svm import SVR
####################################################################################
####################################################################################
####################################################################################
def rmse_log(Y_true,Y_pred):
    temp=np.sqrt(sum( (np.log1p(Y_pred[m]) - np.log1p(Y_true[m]))**2 for m in range(len(Y_true))) / float(len(Y_true)))
    return temp
####################################################################################
####################################################################################
####################################################################################
def xgb_train_mod(y_pow,train_X,Y_dat,valid_X,holdout_X,test_X,param,num_round,log_ind):
    #transform training labels
    if log_ind==1.0:
        train_Y = np.log1p(Y_dat)
    else:
        train_Y = np.power(Y_dat,1.0/y_pow)

    #setup train, validation and holdout inputs to XGB
    xg_train = xgb.DMatrix( train_X, label=train_Y)
    xg_valid = xgb.DMatrix(valid_X)
    xg_holdout = xgb.DMatrix(holdout_X)
    xg_test = xgb.DMatrix(test_X)

    #train model
    watchlist = [ (xg_train,'train') ]
    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50 );
    n_tree = bst.best_iteration

    #predict on validation set, which will be used for fitting the blended model
    pred_XGB_valid = bst.predict( xg_valid, ntree_limit=n_tree )

    #transform predictions
    if log_ind==1.0:
        pred_XGB_valid = np.expm1(pred_XGB_valid)
    else:
        pred_XGB_valid = np.power(pred_XGB_valid,y_pow)

    #predict on holdout set
    pred_XGB_holdout = bst.predict( xg_holdout, ntree_limit=n_tree )

    #transform predictions
    if log_ind==1.0:
        pred_XGB_holdout = np.expm1(pred_XGB_holdout)
    else:
        pred_XGB_holdout = np.power(pred_XGB_holdout,y_pow)

    pred_XGB_test = bst.predict( xg_test, ntree_limit=n_tree )

    #transform predictions
    if log_ind==1.0:
        pred_XGB_test = np.expm1(pred_XGB_test)
    else:
        pred_XGB_test = np.power(pred_XGB_test,y_pow)

    #return predictions
    return [pred_XGB_valid,pred_XGB_holdout,pred_XGB_test]
####################################################################################
####################################################################################
####################################################################################
def split_by_id(n_folds,dfx,assembly_ids):
    #n_holdout: size of holdout set; function will return a set near this size
    #df1: dataframe that will be split
    #assembly_ids: split will be generated using this list of unique ids

    # #dataframe for 
    # holdout_indices=[]
    # n_temp=0
    # while n_temp<n_holdout:
    #     tube_id=np.random.choice(assembly_ids,size=1,replace=False)
    #     assembly_ids=list(set(assembly_ids)-set(tube_id))
    #     holdout_indices=holdout_indices+list(dfx[dfx['tube_assembly_id']==tube_id[0]].index)
    #     n_temp=len(holdout_indices)

    # #generate indices of set used for training
    # train_indices=list(set(dfx.index)-set(holdout_indices))
    # #generate numpy arrays for training features and labels
    # X_dat=dfx.ix[train_indices]
    # Y_dat=dfy.ix[train_indices]
    # holdout_X=dfx.ix[holdout_indices]
    # holdout_Y=dfy.ix[holdout_indices]
    # return [X_dat,Y_dat,holdout_X,holdout_Y]


    size_fold=int(dfx.shape[0]*(1.0/n_folds))
    fold_mat=[]
    for j in range(n_folds):
        n_temp=0
        fold_indices=[]
        while n_temp<size_fold:
            if len(assembly_ids)>0:
                tube_id=np.random.choice(assembly_ids,size=1,replace=False)
                assembly_ids=list(set(assembly_ids)-set(tube_id))
                fold_indices=fold_indices+list(dfx[dfx['tube_assembly_id']==tube_id[0]].index)
                n_temp=len(fold_indices)
            else:
                print "exhausted assembly ids"
                print "length of last fold %i" % n_temp
                n_temp=np.inf
        print n_temp

        train_indices=list(set(dfx.index)-set(fold_indices))
        fold_mat.append((train_indices,fold_indices))
    return fold_mat

####################################################################################
####################################################################################
####################################################################################


dfx=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
dfy=pd.io.parsers.read_table('Y_train.csv',sep=',',header=False)

test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))

indices=[i+1 for i in range(test_X.shape[0])]
assembly_ids=dfx['tube_assembly_id'].unique()


param = {}
param["objective"] = "reg:linear"
param["eval_metric"] = "rmse"
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
num_tree_RF=1000
####################################################################################
####################################################################################

models=[[16.0,0.0],[1.0,0.0],[16.0,1.0]]
n_models=len(models)+1
####################################################################################
####################################################################################
####################################################################################
n_folds=5
fold_indices=split_by_id(n_folds,dfx,assembly_ids)
fold_lengths=np.array([len(x[1]) for x in fold_indices])
dfx.drop("tube_assembly_id",axis=1,inplace=True)
X_dat=np.array(dfx)
Y_dat=np.array(dfy)

rmse_vec=[]
#lists for saving validation, holdout and test predictions
blender_rmse_enet=[]
y_test_enet=np.zeros((test_X.shape[0],n_folds))
#rotate the holdout set
for i in range(n_folds):
# for i in range(1):
    #select holdout set
    holdout_X=X_dat[fold_indices[i][1],:]
    holdout_Y=Y_dat[fold_indices[i][1]]
    #total size of all validation sets to be used
    n_valid_feat=fold_lengths.sum()-len(fold_indices[i][1])
    #matrix for saving predictions on validation sets
    X_valid=np.zeros((n_valid_feat,n_models))
    Y_valid=np.zeros(n_valid_feat)
    #index for saving validation set predictions on each fold
    ind_init_valid=0
    #create a train and validation set from the remaining folds

    #matrix for saving predictions on holdout set
    X_holdout=np.zeros((holdout_X.shape[0],n_models))

    #matrix for saving predictions on test set
    X_test=np.zeros((test_X.shape[0],n_models))
    for j in list(set(range(n_folds))-set([i])):

        #for given fold, generate validation set
        valid_X=X_dat[fold_indices[j][1],:]
        valid_Y=Y_dat[fold_indices[j][1]]

        #given the validation set, train on the remaining indices
        train_indices=[]
        for k in list(set(range(n_folds))-set([i,j])):
            train_indices+=fold_indices[k][1]
        train_X=X_dat[train_indices,:]
        train_Y=Y_dat[train_indices,:]

        #train each model
        X_feat_valid=np.zeros((fold_lengths[j],n_models))
        ind=0
        #iterate over models
        for y_pow,log_ind in models:
            #predict on given validation set using boosted trees
            pred_XGB_valid,pred_XGB_holdout,pred_XGB_test=xgb_train_mod(y_pow,train_X,train_Y,valid_X,holdout_X,test_X,param,num_round,log_ind)
            #save validation set predictions
            X_feat_valid[:,ind]=pred_XGB_valid
            #save holdout and test predictions
            X_holdout[:,ind]+=pred_XGB_holdout
            X_test[:,ind]+=pred_XGB_test
            rmse_vec.append([y_pow,log_ind,rmse_log(valid_Y,pred_XGB_valid),rmse_log(holdout_Y,pred_XGB_holdout)])
            ind+=1
        ####################################################################
        #RF classifier
        RF = RandomForestRegressor(
                n_estimators=num_tree_RF,        #number of trees to generate
                n_jobs=4,               #run in parallel on all cores
                criterion="mse"
                ) 

        print "Training RF model"
        RFmodel = RF.fit(train_X, train_Y.ravel())
        pred_RF_valid=RFmodel.predict(valid_X)
        #save validation set predictions
        X_feat_valid[:,ind]=pred_RF_valid
        pred_RF_holdout=RFmodel.predict(holdout_X)
        pred_RF_test=RFmodel.predict(test_X)
        #save holdout and test predictions
        X_holdout[:,ind]+=pred_RF_holdout
        X_test[:,ind]+=pred_RF_test
        ind+=1
        rmse_vec.append(['RF','RF',rmse_log(valid_Y,pred_RF_valid),rmse_log(holdout_Y,pred_RF_holdout)])
            

        # ####################################################################
        # svr_rbf = SVR(kernel='rbf', C=1.0, gamma=0.1)
        # # svr_lin = SVR(kernel='linear', C=1e3)
        # # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        # print "Training support vector regressor"
        # svr_mod = svr_rbf.fit(train_X, train_Y.ravel())
        # pred_SVR_valid=svr_mod.predict(valid_X)
        # #save validation set predictions
        # X_feat_valid[:,ind]=pred_SVR_valid
        # pred_SVR_holdout=svr_mod.predict(holdout_X)
        # pred_SVR_test=svr_mod.predict(test_X)
        # #save holdout and test predictions
        # X_holdout[:,ind]=pred_SVR_holdout
        # X_test[:,ind]=pred_SVR_test
        # # y_lin = svr_lin.fit(X, y).predict(X)
        # # y_poly = svr_poly.fit(X, y).predict(X)

        # rmse_vec.append(['SVR','SVR',rmse_log(valid_Y,pred_SVR_valid),rmse_log(holdout_Y,pred_SVR_holdout)])
            
        ####################################################################
        #save validation results
        ind_fin_valid=ind_init_valid+fold_lengths[j]
        X_valid[ind_init_valid:ind_fin_valid,:]=X_feat_valid
        Y_valid[ind_init_valid:ind_fin_valid]=valid_Y.ravel()
        ind_init_valid=ind_fin_valid
        ####################################################################


    enet=ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=False, normalize=False, 
        precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, 
        positive=False)
    # LRmodel=LinearRegression(
    #     fit_intercept=False, 
    #     normalize=False, 
    #     copy_X=True)

    # Ridge = RidgeCV(alphas=alphas, normalize=True, cv=5)

    ####################################################################################
    #fit blender using validation set
    enet_mod=enet.fit(X_valid,Y_valid)
    #save model along with holdout set performance measure
    #average over holdout set predictions
    X_holdout/=np.float(n_folds-1)
    pred_holdout=enet_mod.predict(X_holdout)
    #generate OOS performance measure on holdout set
    blender_rmse_enet.append([enet_mod,rmse_log(holdout_Y,pred_holdout)])
    #test set predictions
    y_test_enet[:,i]=enet_mod.predict(X_test)

    print "iteration %g out of %i" %(i+1,n_folds)