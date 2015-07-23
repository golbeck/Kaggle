
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
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
# dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

X_dat=np.loadtxt("X_train_v1.gz",delimiter=",")
Y_dat=np.loadtxt("Y_train.gz",delimiter=",")
n=X_dat.shape[0]

#shuffle train set
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_dat)
np.random.set_state(rng_state)
np.random.shuffle(Y_dat)        

test_X=np.loadtxt("X_test_v1.gz",delimiter=",")
####################################################################################
####################################################################################
####################################################################################
#train model on fraction of data set and validate on left out data
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
param['objective']='reg:linear'
param['eval_metric']='rmse'
param['max_depth'] = 5
param["min_child_weight"] = 1
param["subsample"] = 0.7
param["colsample_bytree"] = 0.8
param["scale_pos_weight"] = 0.8
param['alpha']=0.1
# param['lambda']=0.01
param['eta'] = 0.1
param['silent'] = 0
param['nthread'] = 4
n_class=1
param['num_class'] = n_class
num_round = 100

watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_valid );

print ('prediction error=%f' % (sum( (pred[i] - valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))
valid_gini=Gini(valid_Y, pred)
print 'gini coefficient on validation set=%f' %valid_gini

y_test = bst.predict( xg_test );
df=pd.DataFrame(y_test)
df.columns=['Hazard']
indices=np.loadtxt("X_test_indices.gz",delimiter=",").astype('int32')
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("XGB_predictions.csv",sep=",",index=False)

####################################################################################
####################################################################################
####################################################################################
#train model on full data set
####################################################################################
####################################################################################
####################################################################################
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
# dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

X_dat=np.loadtxt("X_train_v1.gz",delimiter=",")
Y_dat=np.loadtxt("Y_train.gz",delimiter=",")
n=X_dat.shape[0]
test_X=np.loadtxt("X_test_v1.gz",delimiter=",")

#shuffle train set
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_dat)
np.random.set_state(rng_state)
np.random.shuffle(Y_dat)     

sz = X_dat.shape

train_X = X_dat
train_Y = Y_dat

xg_train = xgb.DMatrix( train_X, label=train_Y)
# setup parameters for xgboost
param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['objective']='reg:linear'
param['eval_metric']='rmse'
param['max_depth'] = 5
param["min_child_weight"] = 1
param["subsample"] = 0.7
# params["scale_pos_weight"] = 1.0
param['alpha']=0.1
# param['lambda']=0.01
param['eta'] = 0.1
param['silent'] = 0
param['nthread'] = 4
n_class=1
param['num_class'] = n_class
num_round = 100

watchlist = [ (xg_train,'train') ]
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_train );

print ('prediction error=%f' % (sum( (pred[i] - train_Y[i])**2 for i in range(len(train_Y))) / float(len(train_Y)) ))
valid_gini=Gini(train_Y, pred)
print 'gini coefficient on validation set=%f' %valid_gini

y_test = bst.predict( xg_test );
df=pd.DataFrame(y_test)
df.columns=['Hazard']
indices=np.loadtxt("X_test_indices.gz",delimiter=",").astype('int32')
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("XGB_predictions.csv",sep=",",index=False)

####################################################################################
####################################################################################
####################################################################################
# bag estimates from non-overlapping folds
####################################################################################
####################################################################################
####################################################################################
p=range(20)
frac_=0.05
r0=range(sz[0])
X_folds=np.zeros((len(p)-1,2))
n_test=test_X.shape[0]
y_test_mat=np.zeros((n_test,len(p)-1))
for i in range(len(p)-1):
      r_valid=set(r0[int(p[i]*frac_*sz[0]):int(p[i+1]*frac_*sz[0])])
      r_train=set(r0)-r_valid
      r_valid_indices=[x for x in r_valid]
      r_train_indices=[y for y in r_train]
      train_X = X_dat[r_train_indices, :]
      train_Y = Y_dat[r_train_indices]
      valid_X = X_dat[r_valid_indices, :]
      valid_Y = Y_dat[r_valid_indices]
      xg_train = xgb.DMatrix( train_X, label=train_Y)
      xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
      watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
      bst = xgb.train(param, xg_train, num_round, watchlist );
      pred = bst.predict( xg_valid );
      valid_rmse=np.sqrt(sum( (pred[m] - valid_Y[m])**2 for m in range(len(valid_Y))) / float(len(valid_Y)))
      valid_gini=Gini(valid_Y, pred)
      X_folds[i,0]=valid_rmse
      X_folds[i,1]=valid_gini
      y_test = bst.predict( xg_test );
      y_test_mat[:,i]=y_test
#bag estimates from the model trained on different folds (no need to average since ranking only matter)
y_bag=y_test_mat.sum(axis=1)
print X_folds.mean(axis=0)

df=pd.DataFrame(y_bag)
df.columns=['Hazard']
indices=np.loadtxt("X_test_indices.gz",delimiter=",").astype('int32')
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("XGB_predictions.csv",sep=",",index=False)
####################################################################################
####################################################################################
####################################################################################
# cross validation
####################################################################################
####################################################################################
####################################################################################
depth_grid=[1,3,5,10,20,40]
n_depth=len(depth_grid)
eta_grid=[0.1]
n_eta=len(eta_grid)
lambda_grid=[0.01]
n_lambda=len(lambda_grid)
child_grid=[1,2,3,5,7,10]
n_child=len(child_grid)
subsample_grid=[0.50,0.70,0.90]
n_subsample=len(subsample_grid)

X_cv=np.zeros((n_depth*n_lambda*n_child*n_subsample,6))
ind=0
for i in range(n_depth):
      for j in range(n_lambda):
            for k in range(n_child):
                  for l in range(n_subsample):
                        param = {}
                        param['max_depth']=depth_grid[i]
                        param['eta']=0.1
                        param['lambda']=lambda_grid[j]
                        param['alpha']=0.1
                        param["subsample"] = subsample_grid[l]
                        param['objective']='reg:linear'
                        param['eval_metric']='rmse'
                        param["min_child_weight"] = child_grid[k]
                        # params["scale_pos_weight"] = 1.0
                        param['silent'] = 1
                        param['nthread'] = 4
                        n_class=1
                        param['num_class'] = n_class
                        num_round = num_round_grid[i]

                        watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
                        num_round = 100
                        bst = xgb.train(param, xg_train, num_round, watchlist );
                        # get prediction
                        pred = bst.predict( xg_valid );
                        X_cv[ind,0]=depth_grid[i]
                        X_cv[ind,1]=lambda_grid[j]
                        X_cv[ind,2]=child_grid[k]
                        X_cv[ind,3]=subsample_grid[l]
                        valid_rmse=np.sqrt(sum( (pred[m] - valid_Y[m])**2 for m in range(len(valid_Y))) / float(len(valid_Y)))
                        valid_gini=Gini(valid_Y, pred)
                        X_cv[ind,4]=valid_rmse
                        X_cv[ind,5]=valid_gini
                        ind+=1
                        print ind, valid_rmse, valid_gini

df_cv=pd.DataFrame(X_cv)
df_cv.to_csv('xgboost_cv_v2.csv')

####################################################################################
####################################################################################
####################################################################################
from sklearn import cross_validation

#set up folds and repetitions
n_reps=10
n_folds=5
X_folds=np.zeros((n_reps*n_folds,2))
n_test=test_X.shape[0]
y_test_mat=np.zeros((n_test,n_reps*n_folds))
k_fold = cross_validation.KFold(n=sz[0], n_folds=n_folds)

#cv parameters
depth_grid=[3,5]
n_depth=len(depth_grid)
eta_grid=[0.01,0.1]
n_eta=len(eta_grid)
lambda_grid=[0.01,0.1]
n_lambda=len(lambda_grid)
child_grid=[1,5]
n_child=len(child_grid)
subsample_grid=[0.60,0.70]
n_subsample=len(subsample_grid)
num_round_grid=[50,100,200]
n_num_round_grid=len(num_round_grid)


n_param_1=n_depth
n_param_2=n_num_round_grid
n_param_3=n_child
n_param_4=n_subsample
X_cv=np.zeros((n_param_1*n_param_2*n_param_3*n_param_4,6))
#cv loop
iind=0
for i in range(n_param_1):
    for j in range(n_param_2):
        for k in range(n_param_3):
            for l in range(n_param_4):      
                param_1=depth_grid[i]  
                param_2=num_round_grid[j]
                param_3=subsample_grid[l]
                param_4=child_grid[k]
                param = {}
                param['max_depth']=param_1
                param['eta']=0.1
                # param['lambda']=param_2
                param['alpha']=0.1
                param["subsample"] = param_3
                param['objective']='reg:linear'
                param['eval_metric']='rmse'
                param["min_child_weight"] = param_4
                # params["scale_pos_weight"] = 1.0
                param['silent'] = 1
                param['nthread'] = 4
                n_class=1
                param['num_class'] = n_class
                num_round = param_2

                #cv folds and reps over the parameter set of interest
                X_folds=np.zeros((n_reps*n_folds,2))
                ind=0
                for ii in range(n_reps):   
                    #for each repetition shuffle the data
                    rng_state = np.random.get_state()
                    #randomly permuate the features and outputs using the same shuffle for each epoch
                    np.random.shuffle(X_dat)
                    np.random.set_state(rng_state)
                    np.random.shuffle(Y_dat)   
                    for train_indices, test_indices in k_fold: 
                        train_X = X_dat[train_indices, :]
                        train_Y = Y_dat[train_indices]
                        valid_X = X_dat[test_indices, :]
                        valid_Y = Y_dat[test_indices]
                        xg_train = xgb.DMatrix( train_X, label=train_Y)
                        xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
                        watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
                        bst = xgb.train(param, xg_train, num_round, watchlist );
                        pred = bst.predict( xg_valid );
                        valid_rmse=np.sqrt(sum( (pred[m] - valid_Y[m])**2 for m in range(len(valid_Y))) / float(len(valid_Y)))
                        valid_gini=Gini(valid_Y, pred)
                        X_folds[ind,0]=valid_rmse
                        X_folds[ind,1]=valid_gini
                        ind+=1
                        # y_test = bst.predict( xg_test );
                        # y_test_mat[:,i]=y_test     
                #average cv performance for each parameter set                           
                X_cv[iind,0]=param_1
                X_cv[iind,1]=param_2
                X_cv[iind,2]=param_3
                X_cv[iind,3]=param_4
                temp=X_folds.mean(axis=0)
                valid_rmse=temp[0]
                valid_gini=temp[1]
                X_cv[iind,4]=valid_rmse
                X_cv[iind,5]=valid_gini
                iind+=1

df_cv=pd.DataFrame(X_cv)
df_cv.to_csv('xgboost_cv_v2.csv')
