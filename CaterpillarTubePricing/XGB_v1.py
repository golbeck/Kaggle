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
Y_dat=np.array(pd.io.parsers.read_table('Y_train.csv',sep=',',header=False))
Y_dat=np.log1p(Y_dat)
test_X=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))

indices=[i+1 for i in range(test_X.shape[0])]

n=X_dat.shape[0]
sz = X_dat.shape

# #shuffle train set
# seed=1234
# np.random.seed(seed=seed)
# rng_state = np.random.get_state()
# #randomly permuate the features and outputs using the same shuffle for each epoch
# np.random.shuffle(X_dat)
# np.random.set_state(rng_state)
# np.random.shuffle(Y_dat)        
####################################################################################
####################################################################################
####################################################################################
#train model on fraction of data set and validate on left out data
####################################################################################
####################################################################################
####################################################################################
sz = X_dat.shape

frac=0.95
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
num_round = 7000

watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_valid,ntree_limit=n_tree );
print ('prediction error=%f' % np.sqrt(sum( (pred[i]-valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))

temp=bst.get_fscore()
A=0
for key in temp.keys():
    A+=temp[key]

temp={col_names[int(temp.keys()[i][1:])]:temp.values()[i] for i in range(len(temp))}
feat_imp_sort_XGB = sorted(temp.items(), key=operator.itemgetter(1))


y_test = bst.predict( xg_test, ntree_limit=n_tree );
df=pd.DataFrame(np.expm1(y_test))
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)

####################################################################################
####################################################################################
####################################################################################
#train model on full train set
####################################################################################
####################################################################################
####################################################################################
sz = X_dat.shape

train_X = X_dat
train_Y = Y_dat

xg_train = xgb.DMatrix( train_X, label=train_Y)
# setup parameters for xgboost
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
num_round = 7000

watchlist = [ (xg_train,'train') ]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_valid,ntree_limit=n_tree );
print ('prediction error=%f' % np.sqrt(sum( (pred[i]-valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))

temp=bst.get_fscore()
A=0
for key in temp.keys():
    A+=temp[key]

temp={col_names[int(temp.keys()[i][1:])]:temp.values()[i] for i in range(len(temp))}
feat_imp_sort_XGB = sorted(temp.items(), key=operator.itemgetter(1))


y_test = bst.predict( xg_test, ntree_limit=n_tree );
df=pd.DataFrame(np.expm1(y_test))
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
df.to_csv("XGB_predictions.csv",sep=",",index=False)

####################################################################################
####################################################################################
####################################################################################

sz = df.shape

import itertools
def findsubsets(S,m):
    return set(itertools.combinations(S, m))
    
col_lists=list(findsubsets(col_names,sz[1]-2))
####################################################################################
X_mean=np.zeros((len(col_lists),2))
p=range(4)
frac_=0.25
r0=range(sz[0])

for ii in range(len(col_lists)):
    X_dat=np.array(df[list(col_lists[ii])])
    X_folds=np.zeros((len(p),1))
    n_test=test_X.shape[0]
    for i in p:
        r_valid=set(r0[int(p[i]*frac_*sz[0]):int((p[i]+1)*frac_*sz[0])])
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
        bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50);
        n_tree = bst.best_iteration
        
        pred = bst.predict( xg_valid,ntree_limit=n_tree );
        valid_rmse=np.sqrt(sum( (pred[i]-valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) )
        X_folds[i,0]=valid_rmse
    X_mean[ii,0]=ii
    X_mean[ii,1:]=X_folds.mean(axis=0)
    print "iteration %i out of %g" %(ii+1,len(col_lists))
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
X_folds=np.zeros((len(p),1))
n_test=test_X.shape[0]
y_test_mat=np.zeros((n_test,len(p)))
# least_imp=[]
# feat_imp_mat=np.zeros((sz[1],len(p)))
for i in p:
    r_valid=set(r0[int(p[i]*frac_*sz[0]):int((p[i]+1)*frac_*sz[0])])
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
    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50);
    n_tree = bst.best_iteration
    pred = bst.predict( xg_valid,ntree_limit=n_tree );
    valid_rmse=np.sqrt(sum( (pred[i]-valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) )
    X_folds[i,0]=valid_rmse
    y_test = bst.predict( xg_test, ntree_limit=n_tree);
    y_test_mat[:,i]=np.expm1(y_test)
    # feat_imp = bst.get_fscore()
    # temp = np.array(feat_imp.values()).argmin()
    # least_imp.append(feat_imp.keys()[temp])


    print "iteration %i out of %g" %(i+1,len(p))
    # A=0
    # for key in feat_imp.keys():
    #   A+=feat_imp[key]
    # feat_imp_mat[:,i]=np.array(feat_imp.values(),dtype=np.float)/A
#bag estimates from the model trained on different folds (no need to average since ranking only matter)
# weights=np.max(X_folds)-X_folds
# weights/=weights.sum()
# y_bag=np.array([weights[i]*y_test_mat[:,i] for i in range(len(weights))]).transpose()
y_bag=y_test_mat.mean(axis=1)
print X_folds.mean(axis=0)

# for i in range(5):
#     ind_min= feat_imp_mat.argsort(axis=0)[i,:]
#     keys_ind = [feat_imp.keys()[i] for i in ind_min]
#     keys_ind0 = [np.int(keys_ind[i][1:]) for i in range(len(keys_ind))]
#     print [col_names[i] for i in keys_ind0]

df=pd.DataFrame(y_bag)
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("XGB_predictions.csv",sep=",",index=False)
####################################################################################
####################################################################################
####################################################################################
# bag estimates from non-overlapping folds, and look at holdout set performance
####################################################################################
####################################################################################
####################################################################################
frac=0.05
r0=range(sz[0])
r_holdout=set(r0[0:int(frac*sz[0])])
r_train=set(r0)-r_holdout
holdout_indices=[x for x in r_holdout]
train_indices=[y for y in r_train]
holdout_X = X_dat[holdout_indices, :]
holdout_Y = Y_dat[holdout_indices]

xg_holdout = xgb.DMatrix( holdout_X, label=holdout_Y)

X_cv_train = X_dat[train_indices, :]
Y_cv_train = Y_dat[train_indices]

p=range(20)
frac_=0.05
sz=X_cv_train.shape
r0=range(sz[0])
X_folds=np.zeros((len(p),2))
n_test=holdout_X.shape[0]
y_holdout_mat=np.zeros((n_test,len(p)))
# least_imp=[]
# feat_imp_mat=np.zeros((sz[1],len(p)))
for i in p:
    r_valid=set(r0[int(p[i]*frac_*sz[0]):int((p[i]+1)*frac_*sz[0])])
    r_train=set(r0)-r_valid
    r_valid_indices=[x for x in r_valid]
    r_train_indices=[y for y in r_train]
    train_X = X_cv_train[r_train_indices, :]
    train_Y = Y_cv_train[r_train_indices]
    valid_X = X_cv_train[r_valid_indices, :]
    valid_Y = Y_cv_train[r_valid_indices]
    xg_train = xgb.DMatrix( train_X, label=train_Y)
    xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
    watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50);
    n_tree = bst.best_iteration
    pred = bst.predict( xg_valid,ntree_limit=n_tree );
    valid_rmse=np.sqrt(sum( (pred[i]-valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) )
    X_folds[i,0]=valid_rmse
    y_holdout = bst.predict( xg_holdout, ntree_limit=n_tree);
    y_holdout_mat[:,i]=np.expm1(y_holdout)
    holdout_rmse=np.sqrt(sum( (y_holdout[i]-holdout_Y[i])**2 for i in range(len(holdout_Y))) / float(len(holdout_Y)) )
    X_folds[i,1]=holdout_rmse
    # feat_imp = bst.get_fscore()
    # temp = np.array(feat_imp.values()).argmin()
    # least_imp.append(feat_imp.keys()[temp])


    print "iteration %i out of %g" %(i+1,len(p))
    # A=0
    # for key in feat_imp.keys():
    #   A+=feat_imp[key]
    # feat_imp_mat[:,i]=np.array(feat_imp.values(),dtype=np.float)/A
#bag estimates from the model trained on different folds (no need to average since ranking only matter)
# weights=np.max(X_folds)-X_folds
# weights/=weights.sum()
# y_bag=np.array([weights[i]*y_test_mat[:,i] for i in range(len(weights))]).transpose()
y_bag=y_holdout_mat.mean(axis=1)
holdout_rmse=np.sqrt(sum( (y_bag[i]-holdout_Y[i])**2 for i in range(len(holdout_Y))) / float(len(holdout_Y)) )
print X_folds.mean(axis=0)
####################################################################################
####################################################################################
####################################################################################
# cross validation
####################################################################################
####################################################################################
####################################################################################
from sklearn import cross_validation

#set up folds and repetitions
n_reps=1
n_folds=4
X_folds=np.zeros((n_reps*n_folds,2))
n_test=test_X.shape[0]
y_test_mat=np.zeros((n_test,n_reps*n_folds))
k_fold = cross_validation.KFold(n=sz[0], n_folds=n_folds,random_state=seed)

#cv parameters
depth_grid=[25]
n_depth=len(depth_grid)
gamma_grid=[0.0]
n_gamma_grid=len(gamma_grid)
child_grid=[15]
n_child=len(child_grid)
subsample_grid=[0.87]
n_subsample=len(subsample_grid)
colsample_grid=[0.50]
n_colsample=len(colsample_grid)

n_param_1=n_depth
n_param_2=n_gamma_grid
n_param_3=n_child
n_param_4=n_subsample
n_param_5=n_colsample
X_cv=np.zeros((n_param_1*n_param_2*n_param_3*n_param_4*n_param_5,6))
least_imp=[]
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
                    # param['lambda']=param_2
                    # param['alpha']=0.1
                    param['gamma']=param_2
                    param["subsample"] = param_4
                    param["colsample_bytree"] = param_5
                    param['objective']='reg:linear'
                    param['eval_metric']='rmse'
                    param["min_child_weight"] = param_3
                    # param["scale_pos_weight"] = 1.0
                    param['silent'] = 1
                    param['nthread'] = 4
                    n_class=1
                    param['num_class'] = n_class
                    num_round = 1000

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
                            bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50);
                            n_tree = bst.best_iteration
                            feat_imp = bst.get_fscore()
                            temp = np.array(feat_imp.values()).argmin()
                            least_imp.append(feat_imp.keys()[temp])
                            pred = bst.predict( xg_valid, ntree_limit=n_tree);                            
                            valid_rmse=np.sqrt(sum( (pred[m]-valid_Y[m])**2 for m in range(len(valid_Y))) / float(len(valid_Y)) )
                            X_folds[ind,0]=valid_rmse
                            ind+=1
                            # y_test = bst.predict( xg_test );
                            # y_test_mat[:,i]=y_test     
                    #average cv performance for each parameter set                           
                    X_cv[iind,0]=param_1
                    X_cv[iind,1]=param_2
                    X_cv[iind,2]=param_3
                    X_cv[iind,3]=param_4
                    X_cv[iind,4]=param_5
                    temp=X_folds.mean(axis=0)
                    valid_rmse=temp[0]
                    valid_gini=temp[1]
                    X_cv[iind,5]=valid_rmse
                    iind+=1
                    print "iteration %i out of %g" % (iind, X_cv.shape[0])

df_cv=pd.DataFrame(X_cv)
df_cv.to_csv('xgboost_cv_v2.csv')
