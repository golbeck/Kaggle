
import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from sklearn.decomposition import PCA
from sklearn import preprocessing

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
# dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

dat=pd.io.parsers.read_table('train.csv',sep=',',header=0)


dat.drop('T2_V8',axis=1,inplace=True)
dat.drop('T2_V12',axis=1,inplace=True)
dat.drop('T2_V11',axis=1,inplace=True)
dat.drop('T1_V17',axis=1,inplace=True)
dat.drop('T2_V3',axis=1,inplace=True)
#convert str levels to numerical levels

df_pca=pd.DataFrame()
df1=pd.DataFrame()
scaler_list=[]
for col in dat.columns[2:]:
    if type(dat[col].ix[0])!=str:
        #normalize to [0,1]
        df_pca[col] = preprocessing.scale(np.array(dat[col].astype('float')))
        scaler_list.append(preprocessing.StandardScaler().fit(np.array(dat[col].astype('float'))))
    else:
        df1[col]=pd.Categorical(dat[col]).labels

n=df_pca.shape[1]
pca = PCA(n_components=n)
pca_fit=pca.fit(df_pca)
df_fit=pd.DataFrame(pca_fit.transform(df_pca),columns=df_pca.columns)

df= pd.concat([df_fit, df1], axis=1)

col_names=df.columns

X_dat=np.array(df)
Y_dat=np.array(dat['Hazard'])
del dat, df_pca, df1, df_fit, df

####################################################################################
dat=pd.io.parsers.read_table('test.csv',sep=',',header=0)

dat.drop('T2_V8',axis=1,inplace=True)
dat.drop('T2_V12',axis=1,inplace=True)
dat.drop('T2_V11',axis=1,inplace=True)
dat.drop('T1_V17',axis=1,inplace=True)
dat.drop('T2_V3',axis=1,inplace=True)

indices=dat['Id']
#convert str levels to numerical levels
df_pca=pd.DataFrame()
df1=pd.DataFrame()
ind=0
for col in dat.columns[1:]:
    if type(dat[col].ix[0])!=str:
        #normalize to [0,1]
        df_pca[col] = scaler_list[ind].transform(np.array(dat[col].astype('float')))
        ind+=1
    else:        
        df1[col]=pd.Categorical(dat[col]).labels

df_fit=pd.DataFrame(pca_fit.transform(df_pca),columns=df_pca.columns)
df= pd.concat([df_fit, df1], axis=1)

test_X=np.array(df)
del dat, df_pca, df1, df_fit, df

####################################################################################
n=X_dat.shape[0]
sz = X_dat.shape

#shuffle train set
seed=1234
np.random.seed(seed=seed)
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_dat)
np.random.set_state(rng_state)
np.random.shuffle(Y_dat)        
####################################################################################
####################################################################################
####################################################################################
#train model on fraction of data set and validate on left out data
####################################################################################
####################################################################################
####################################################################################
frac=0.90
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
param["min_child_weight"] = 20
param["subsample"] = 0.80
param["colsample_bytree"] = 0.80
param["scale_pos_weight"] = 1.0
# param['gamma'] = 5
param["silent"] = 1
param["max_depth"] = 7
param['nthread'] = 4
n_class=1
param['num_class'] = n_class
num_round = 7000

watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
bst = xgb.train(param, xg_train, num_round, watchlist ,early_stopping_rounds=50);
print bst.get_fscore()
# get prediction
pred = bst.predict( xg_valid );

print ('prediction error=%f' % (sum( (pred[i] - valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))
valid_gini=Gini(valid_Y, pred)
print 'gini coefficient on validation set=%f' %valid_gini

temp=bst.get_fscore()
A=0
for key in temp.keys():
    A+=temp[key]



    

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
# param["scale_pos_weight"] = 1.0
param['alpha']=0.1
# param['lambda']=0.01
param['eta'] = 0.1
param['silent'] = 0
param['nthread'] = 4
n_class=1
param['num_class'] = n_class
num_round = 100

watchlist = [ (xg_train,'train') ]
bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=50 );
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
p=range(10)
frac_=0.10
r0=range(sz[0])
X_folds=np.zeros((len(p),2))
n_test=test_X.shape[0]
y_test_mat=np.zeros((n_test,len(p)))
least_imp=[]
feat_imp_mat=np.zeros((sz[1],len(p)))
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
    pred = bst.predict( xg_valid );
    feat_imp = bst.get_fscore()
    temp = np.array(feat_imp.values()).argmin()
    least_imp.append(feat_imp.keys()[temp])


    A=0
    for key in feat_imp.keys():
      A+=feat_imp[key]
    feat_imp_mat[:,i]=np.array(feat_imp.values(),dtype=np.float)/A

    valid_rmse=np.sqrt(sum( (pred[m] - valid_Y[m])**2 for m in range(len(valid_Y))) / float(len(valid_Y)))
    valid_gini=Gini(valid_Y, pred)
    X_folds[i,0]=valid_rmse
    X_folds[i,1]=valid_gini
    y_test = bst.predict( xg_test );
    y_test_mat[:,i]=y_test
#bag estimates from the model trained on different folds (no need to average since ranking only matter)
y_bag=y_test_mat.sum(axis=1)
print X_folds.mean(axis=0)

for i in range(5):
    ind_min= feat_imp_mat.argsort(axis=0)[i,:]
    keys_ind = [feat_imp.keys()[i] for i in ind_min]
    keys_ind0 = [np.int(keys_ind[i][1:]) for i in range(len(keys_ind))]
    print [col_names[i] for i in keys_ind0]

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
                        # param["scale_pos_weight"] = 1.0
                        param['silent'] = 1
                        param['nthread'] = 4
                        n_class=1
                        param['num_class'] = n_class
                        num_round = num_round_grid[i]

                        watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
                        num_round = 100
                        bst = xgb.train(param, xg_train, num_round, watchlist,early_stopping_rounds=50);
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
n_reps=2
n_folds=2
X_folds=np.zeros((n_reps*n_folds,2))
n_test=test_X.shape[0]
y_test_mat=np.zeros((n_test,n_reps*n_folds))
k_fold = cross_validation.KFold(n=sz[0], n_folds=n_folds)

#cv parameters
depth_grid=[7,10,13]
n_depth=len(depth_grid)
eta_grid=[0.01,0.1]
n_eta=len(eta_grid)
lambda_grid=[0.01,0.1]
n_lambda=len(lambda_grid)
child_grid=[20,25,30]
n_child=len(child_grid)
subsample_grid=[0.50,0.80]
n_subsample=len(subsample_grid)
num_round_grid=[50,100,200]
n_num_round_grid=len(num_round_grid)
gamma_grid=[0.0]
n_gamma_grid=len(gamma_grid)
colsample_grid=[0.80,0.85,0.90]
n_colsample=len(colsample_grid)

n_param_1=n_depth
n_param_2=n_gamma_grid
n_param_3=n_child
n_param_4=n_subsample
n_param_5=n_colsample
X_cv=np.zeros((n_param_1*n_param_2*n_param_3*n_param_4*n_param_5,7))
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
                    num_round = 3000

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
                    X_cv[iind,4]=param_5
                    temp=X_folds.mean(axis=0)
                    valid_rmse=temp[0]
                    valid_gini=temp[1]
                    X_cv[iind,5]=valid_rmse
                    X_cv[iind,6]=valid_gini
                    iind+=1
                    print "iteration %i out of %g" % (iind, X_cv.shape[0])

df_cv=pd.DataFrame(X_cv)
df_cv.to_csv('xgboost_cv_v2.csv')
