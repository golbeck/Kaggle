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
def xgb_bagged(m,y_pow,X_dat,Y_dat,param,num_round,log_ind):
    p=range(m)
    frac=1.0/m
    sz=X_dat.shape
    r0=range(sz[0])
    X_folds=np.zeros((len(p),1))
    least_imp=[]
    feat_imp_mat=np.zeros((sz[1],len(p)))
    for i in p:
        #randomly select without replacement the validation set for each fold
        r_valid=set(np.random.choice(range(sz[0]),size=int(frac*sz[0]),replace=False))
        # r_valid=set(r0[int(p[i]*frac*sz[0]):int((p[i]+1)*frac*sz[0])])
        r_train=set(r0)-r_valid
        r_valid_indices=[x for x in r_valid]
        r_train_indices=[y for y in r_train]
        train_X = X_dat[r_train_indices, :]
        valid_X = X_dat[r_valid_indices, :]
        if (log_ind==1):
            #take log     
            valid_Y=np.log1p(Y_dat[r_valid_indices])
            train_Y=np.log1p(Y_dat[r_train_indices])
        else:
            valid_Y = np.power(Y_dat[r_valid_indices],1.0/y_pow)
            train_Y = np.power(Y_dat[r_train_indices],1.0/y_pow)
        xg_train = xgb.DMatrix( train_X, label=train_Y)
        xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
        watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
        bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50);
        n_tree = bst.best_iteration
        pred = bst.predict( xg_valid, ntree_limit=n_tree );
        if (log_ind==1):
            #take log     
            valid_Y=np.expm1(valid_Y)
            pred = np.expm1(pred)
        else:
            valid_Y = np.power(valid_Y,y_pow)
            pred = np.power(pred,y_pow)

        valid_rmse=np.sqrt(sum( (np.log1p(pred[m]) - np.log1p(valid_Y[m]))**2 for m in range(len(valid_Y))) / float(len(valid_Y)))
        X_folds[i,0]=valid_rmse

    print X_folds.mean(axis=0)
    return X_folds.mean(axis=0)
####################################################################################
####################################################################################
####################################################################################
def xgb_bagged_holdout(m,y_pow,X_dat,Y_dat,holdout_X,holdout_Y,param,num_round,log_ind):
    p=range(m)
    frac=1.0/m
    sz=X_dat.shape
    r0=range(sz[0])
    X_folds=np.zeros((len(p),2))
    for i in p:
        #holdout set
        if (log_ind==1):
            #take log     
            holdout_Y=np.log1p(holdout_Y)
        else:
            holdout_Y = np.power(holdout_Y,1.0/y_pow)
        xg_holdout = xgb.DMatrix(holdout_X, label=holdout_Y)
        #randomly select without replacement the validation set for each fold
        r_valid=set(np.random.choice(range(sz[0]),size=int(frac*sz[0]),replace=False))
        # r_valid=set(r0[int(p[i]*frac*sz[0]):int((p[i]+1)*frac*sz[0])])
        r_train=set(r0)-r_valid
        r_valid_indices=[x for x in r_valid]
        r_train_indices=[y for y in r_train]
        train_X = X_dat[r_train_indices, :]
        valid_X = X_dat[r_valid_indices, :]
        if (log_ind==1):
            #take log     
            valid_Y=np.log1p(Y_dat[r_valid_indices])
            train_Y=np.log1p(Y_dat[r_train_indices])
        else:
            valid_Y = np.power(Y_dat[r_valid_indices],1.0/y_pow)
            train_Y = np.power(Y_dat[r_train_indices],1.0/y_pow)
        xg_train = xgb.DMatrix( train_X, label=train_Y)
        xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
        watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
        bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50);
        n_tree = bst.best_iteration
        #validation set
        pred = bst.predict( xg_valid, ntree_limit=n_tree );
        if (log_ind==1):
            #take log     
            valid_Y=np.expm1(valid_Y)
            pred = np.expm1(pred)
        else:
            valid_Y = np.power(valid_Y,y_pow)
            pred = np.power(pred,y_pow)

        valid_rmse=np.sqrt(sum( (np.log1p(pred[m]) - np.log1p(valid_Y[m]))**2 for m in range(len(valid_Y))) / float(len(valid_Y)))
        X_folds[i,0]=valid_rmse

        #holdout set

        pred = bst.predict( xg_holdout, ntree_limit=n_tree );
        if (log_ind==1):
            #take log     
            holdout_Y=np.expm1(holdout_Y)
            pred = np.expm1(pred)
        else:
            holdout_Y = np.power(holdout_Y,y_pow)
            pred = np.power(pred,y_pow)

        holdout_rmse=np.sqrt(sum( (np.log1p(pred[m]) - np.log1p(holdout_Y[m]))**2 for m in range(len(holdout_Y))) / float(len(holdout_Y)))
        X_folds[i,1]=holdout_rmse


    print X_folds.mean(axis=0)
    return X_folds.mean(axis=0)
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
test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))

indices=[i+1 for i in range(test_X.shape[0])]

n=X_dat.shape[0]
sz = X_dat.shape
Y_dat=np.array(pd.io.parsers.read_table('Y_train.csv',sep=',',header=False))
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
num_round = 2000
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
valid_Y=np.power(valid_Y,y_pow)
print ('prediction error=%f' % np.sqrt(sum( (np.log1p(pred1[i])-np.log1p(valid_Y[i]))**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))

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

print ('prediction error=%f' % np.sqrt(sum( ((pred[i])-(valid_Y[i]))**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))

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
num_round = 7000
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
num_round = 7000
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
num_round = 2000
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
pred3 = np.power(pred,y_pow)
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
pred4 = pred
# pred2 = np.expm1(pred)


preds=0.4*(pred1)+0.1*np.expm1(pred2)+0.4*pred3+0.1*np.expm1(pred4)
df=pd.DataFrame(preds)
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)



####################################################################################
####################################################################################
####################################################################################
# cross validation
####################################################################################
####################################################################################
####################################################################################
#cv parameters
depth_grid=[6,8,10]
n_depth=len(depth_grid)
# eta_grid=[0.01,0.1]
# n_eta=len(eta_grid)
# lambda_grid=[0.01,0.1]
# n_lambda=len(lambda_grid)
child_grid=[5,10,15]
n_child=len(child_grid)
subsample_grid=[0.50,0.65,0.80]
n_subsample=len(subsample_grid)
# num_round_grid=[50,100,200]
# n_num_round_grid=len(num_round_grid)
gamma_grid=[0.0]
n_gamma_grid=len(gamma_grid)
colsample_grid=[0.75,0.85]
n_colsample=len(colsample_grid)

n_param_1=n_depth
n_param_2=n_gamma_grid
n_param_3=n_child
n_param_4=n_subsample
n_param_5=n_colsample
X_cv=np.zeros((n_param_1*n_param_2*n_param_3*n_param_4*n_param_5,6))
#cv loop
iind=0
for i in range(n_param_1):
    for j in range(n_param_2):
        for k in range(n_param_3):
            for l in range(n_param_4):     
                for ll in range(n_param_5):   
                    param_1=depth_grid[i]  
                    param_2=gamma_grid[j]
                    param_3=child_grid[k]
                    param_4=subsample_grid[l]
                    param_5=colsample_grid[ll]
                    param = {}
                    param['max_depth']=param_1
                    param['eta']=0.01
                    param['gamma']=param_2
                    param["subsample"] = param_4
                    param["colsample_bytree"] = param_5
                    param['objective']='reg:linear'
                    param['eval_metric']='rmse'
                    param["min_child_weight"] = param_3
                    param['silent'] = 1
                    param['nthread'] = 4
                    param['num_class'] = 1
                    num_round = 2000

                    #average cv performance for each parameter set                           
                    X_cv[iind,0]=param_1
                    X_cv[iind,1]=param_2
                    X_cv[iind,2]=param_3
                    X_cv[iind,3]=param_4
                    X_cv[iind,4]=param_5
                    y_pow=1.0
                    log_ind=1.0
                    m=4
                    X_cv[iind,5]=xgb_bagged(m,y_pow,test_X,X_dat,param,num_round,log_ind)
                    iind+=1
                    print "iteration %i out of %g" % (iind, X_cv.shape[0])
                    print X_cv[:,5]

df_cv=pd.DataFrame(X_cv)
df_cv.to_csv('xgboost_cv_v2.csv')

####################################################################################
####################################################################################
####################################################################################
# cross validation
####################################################################################
####################################################################################
####################################################################################

#cv parameters
depth_grid=[10,15]
n_depth=len(depth_grid)
# eta_grid=[0.01,0.1]
# n_eta=len(eta_grid)
# lambda_grid=[0.01,0.1]
# n_lambda=len(lambda_grid)
child_grid=[5]
n_child=len(child_grid)
subsample_grid=[0.50,0.65,0.80]
n_subsample=len(subsample_grid)
# num_round_grid=[50,100,200]
# n_num_round_grid=len(num_round_grid)
gamma_grid=[0.0]
n_gamma_grid=len(gamma_grid)
colsample_grid=[0.75,0.85]
n_colsample=len(colsample_grid)

n_param_1=n_depth
n_param_2=n_gamma_grid
n_param_3=n_child
n_param_4=n_subsample
n_param_5=n_colsample

holdout_sets=4
X_cv=np.zeros((n_param_1*n_param_2*n_param_3*n_param_4*n_param_5*holdout_sets,7))
#cv loop
iind=0
for ii in range(holdout_sets):
    df1=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
    df_Y=pd.io.parsers.read_table('Y_train.csv',sep=',',header=False)

    frac=0.05
    sz=df1.shape
    n_holdout=int(frac*sz[0])
    n_temp=0
    assembly_ids=df1['tube_assembly_id'].unique()
    temp=pd.DataFrame()
    while n_temp<n_holdout:
        tube_id=np.random.choice(assembly_ids,size=1,replace=False)
        assembly_ids=list(set(assembly_ids)-set(tube_id))
        if temp.empty==True:
            temp=df1[df1['tube_assembly_id']==tube_id[0]]
        else:
            temp=temp.append(df1[df1['tube_assembly_id']==tube_id[0]])
        n_temp=temp.shape[0]


    holdout_indices=temp.index
    train_indices=list(set(df1.index)-set(holdout_indices))
    df1.drop('tube_assembly_id',axis=1, inplace=True)
    X_dat=np.array(df1.ix[train_indices])
    Y_dat=np.array(df_Y.ix[train_indices])
    temp.drop('tube_assembly_id',axis=1,inplace=True)
    holdout_X=np.array(temp)
    holdout_Y=np.array(df_Y.ix[holdout_indices])


    for i in range(n_param_1):
        for j in range(n_param_2):
            for k in range(n_param_3):
                for l in range(n_param_4):     
                    for ll in range(n_param_5):   
                        param_1=depth_grid[i]  
                        param_2=gamma_grid[j]
                        param_3=child_grid[k]
                        param_4=subsample_grid[l]
                        param_5=colsample_grid[ll]
                        param = {}
                        param['max_depth']=param_1
                        param['eta']=0.01
                        param['gamma']=param_2
                        param["subsample"] = param_4
                        param["colsample_bytree"] = param_5
                        param['objective']='reg:linear'
                        param['eval_metric']='rmse'
                        param["min_child_weight"] = param_3
                        param['silent'] = 1
                        param['nthread'] = 4
                        param['num_class'] = 1
                        num_round = 1000

                        #average cv performance for each parameter set                           
                        X_cv[iind,0]=param_1
                        X_cv[iind,1]=param_2
                        X_cv[iind,2]=param_3
                        X_cv[iind,3]=param_4
                        X_cv[iind,4]=param_5
                        y_pow=1.0
                        log_ind=1.0
                        m=4
                        rmse_temp=xgb_bagged_holdout(m,y_pow,X_dat,Y_dat,holdout_X,holdout_Y,param,num_round,log_ind)
                        X_cv[iind,5]=rmse_temp[0]
                        X_cv[iind,6]=rmse_temp[1]
                        iind+=1
                        print "iteration %i out of %g" % (iind, X_cv.shape[0])
                        print X_cv[:,5:]

df_cv=pd.DataFrame(X_cv)
df_cv.to_csv('xgboost_cv_v3.csv')