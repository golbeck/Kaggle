
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
def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

####################################################################################
####################################################################################
####################################################################################
def xgb_train_full(y_pow,train_X,Y_dat,test_X,param,num_round):

    train_Y = np.power(Y_dat,1.0/y_pow)
    xg_train = xgb.DMatrix( train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X)
    watchlist = [ (xg_train,'train') ]
    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50 );
    n_tree = bst.best_iteration
    pred = bst.predict( xg_test, ntree_limit=n_tree );
    pred = np.power(pred,y_pow)
    return pred
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


    #predict on holdout set
    pred_test = bst.predict( xg_test, ntree_limit=n_tree )

    #transform predictions
    if log_ind==1.0:
        pred_test = np.expm1(pred_test)
    else:
        pred_test = np.power(pred_test,y_pow)

    #return predictions
    return [pred_valid,pred_holdout,pred_test]
####################################################################################
####################################################################################
####################################################################################
# pwd_temp=os.getcwd()
# dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
# # dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
# dir1=dir1+'/data' 
# if pwd_temp!=dir1:
#     os.chdir(dir1)

dat=pd.io.parsers.read_table('train.csv',sep=',',header=0)

#convert str levels to numerical levels
df=pd.DataFrame()
for col in dat.columns[2:]:
    if type(dat[col].ix[0])==str:
        df[col]=pd.Categorical(dat[col]).labels
    else:
        df[col]=dat[col]


df.drop('T2_V10', axis=1, inplace=True)
df.drop('T1_V13', axis=1, inplace=True)
df.drop('T2_V5', axis=1, inplace=True)
# df.drop('T2_V12',axis=1,inplace=True)
# df.drop('T2_V10', axis=1, inplace=True)
# df.drop('T2_V7', axis=1, inplace=True)
# df.drop('T1_V13', axis=1, inplace=True)
# df.drop('T1_V10', axis=1, inplace=True)

# df.drop('T2_V8',axis=1,inplace=True)
# df.drop('T2_V12',axis=1,inplace=True)
# df.drop('T2_V11',axis=1,inplace=True)
# df.drop('T1_V17',axis=1,inplace=True)
# df.drop('T2_V3',axis=1,inplace=True)
# df.drop('T1_V17',axis=1,inplace=True)


X_dat=np.array(df)
Y_dat=np.array(dat['Hazard'])
del dat, df

dat=pd.io.parsers.read_table('test.csv',sep=',',header=0)

indices=dat['Id']
#convert str levels to numerical levels
df=pd.DataFrame()
for col in dat.columns[1:]:
    if type(dat[col].ix[0])==str:
        df[col]=pd.Categorical(dat[col]).labels
    else:
        df[col]=dat[col]


df.drop('T2_V10', axis=1, inplace=True)
df.drop('T1_V13', axis=1, inplace=True)
df.drop('T2_V5', axis=1, inplace=True)
# df.drop('T2_V12',axis=1,inplace=True)
# df.drop('T2_V10', axis=1, inplace=True)
# df.drop('T2_V7', axis=1, inplace=True)
# df.drop('T1_V13', axis=1, inplace=True)
# df.drop('T1_V10', axis=1, inplace=True)

# df.drop('T1_V17',axis=1,inplace=True)

# df.drop('T2_V8',axis=1,inplace=True)
# df.drop('T2_V12',axis=1,inplace=True)
# df.drop('T2_V11',axis=1,inplace=True)
# df.drop('T1_V17',axis=1,inplace=True)
# df.drop('T2_V3',axis=1,inplace=True)
# df.drop('T1_V17',axis=1,inplace=True)

col_names=df.columns

test_X=np.array(df)
del dat, df

n=X_dat.shape[0]
sz = X_dat.shape

     
# setup parameters for xgboost
param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softmax'
# scale weight of positive examples

param["objective"] = "reg:linear" 
param["eta"] = 0.01 
param["min_child_weight"] = 25 
param["subsample"] = 0.8 
param["colsample_bytree"] = 0.85 
param["scale_pos_weight"] = 1.0 
param["silent"] = 1 
param["max_depth"] = 10 
param['nthread'] = 4 
param['num_class'] = 1 
num_round = 7000

####################################################################################
####################################################################################
####################################################################################
# run m-folds CV to determine OOS performance of each model
####################################################################################
####################################################################################
####################################################################################
#shuffle train set
seed=3456
np.random.seed(seed=seed)
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_dat)
np.random.set_state(rng_state)
np.random.shuffle(Y_dat)   
sz=X_dat.shape

#generate train, validation, and holdout set
frac_valid=0.05
frac_holdout=0.05
frac=1.0-frac_valid-frac_holdout
n_train=int(frac*sz[0])
n_valid=int((1.0-frac_holdout)*sz[0])
#train set for fitting regressors
train_X=X_dat[0:n_train,:]
train_Y=Y_dat[0:n_train]
#validation set for fitting the blend
valid_X=X_dat[n_train:n_valid,:]
valid_Y=Y_dat[n_train:n_valid]
#holdout set for testing the blend
holdout_X=X_dat[n_valid:,:]
holdout_Y=Y_dat[n_valid:]

y_pow_vec=[2.0,3.0,8.0]
n_pow=len(y_pow_vec)
X_folds=np.zeros((n_pow,3))
num_round=7000
log_ind=0.0
X_net_valid=np.zeros((valid_X.shape[0],n_pow+1))
X_net_holdout=np.zeros((holdout_X.shape[0],n_pow+1))
X_net_test=np.zeros((test_X.shape[0],n_pow+1))
for i in range(n_pow):
    y_pow=y_pow_vec[i]
    temp=xgb_train_mod(y_pow,train_X,train_Y,valid_X,holdout_X,test_X,param,num_round,log_ind)
    X_net_valid[:,i]=temp[0]
    X_net_holdout[:,i]=temp[1]
    X_net_test[:,i]=temp[2]

log_ind=1.0
temp=xgb_train_mod(y_pow,train_X,train_Y,valid_X,holdout_X,test_X,param,num_round,log_ind)
X_net_valid[:,n_pow]=temp[0]
X_net_holdout[:,n_pow]=temp[1]
X_net_test[:,n_pow]=temp[2]
####################################################################################
####################################################################################
####################################################################################
#Elastic net blender
####################################################################################
####################################################################################
####################################################################################
from sklearn.linear_model import ElasticNet
# objective function: 1 / (2 * n_samples) * ||y - Xw||^2_2 +
# + alpha * l1_ratio * ||w||_1
# + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

enet=ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=False, normalize=False, 
    precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, 
    positive=False)

enet_mod=enet.fit(X_net_valid,valid_Y)
pred_holdout=enet_mod.predict(X_net_holdout)

holdout_gini=Gini(holdout_Y,pred_holdout)

valid_rmse=np.sqrt(sum( (pred_holdout[m] - holdout_Y[m])**2 for m in range(len(holdout_Y))) / float(len(holdout_Y)))
print valid_rmse, holdout_gini                     

pred_test=enet_mod.predict(X_net_test)

df=pd.DataFrame(pred_test)
df.columns=['Hazard']
indices=np.loadtxt("X_test_indices.gz",delimiter=",").astype('int32')
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)
####################################################################################
####################################################################################
####################################################################################
#Linear regression blender
####################################################################################
####################################################################################
####################################################################################
#model
from sklearn.linear_model import LinearRegression
LRmodel=LinearRegression(
    fit_intercept=False, 
    normalize=False, 
    copy_X=True
    )

#train
LRmodel = LRmodel.fit(X_net_valid,valid_Y)
#predict 
pred_holdout=LRmodel.predict(X_net_holdout)
holdout_gini=Gini(holdout_Y,pred_holdout)

valid_rmse=np.sqrt(sum( (pred_holdout[m] - holdout_Y[m])**2 for m in range(len(holdout_Y))) / float(len(holdout_Y)))
print valid_rmse, holdout_gini      