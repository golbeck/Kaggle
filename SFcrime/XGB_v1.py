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
def evalerror(pred, y_in):
    error=sum( pred[i] != y_in[i] for i in range(len(pred))) / float(len(pred))
    return error

####################################################################################
####################################################################################
####################################################################################
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/SFcrime'
# dir1='/home/golbeck/Workspace/Kaggle/SFcrime'
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
param['objective'] = 'multi:softprob'
# param['eval_metric']='rmse'
param["min_child_weight"] = 5
param["subsample"] = 0.7
# params["scale_pos_weight"] = 1.0
param['alpha']=0.1
param['eta'] = 0.1
# param['max_depth'] = 5
param['silent'] = 0
param['nthread'] = 4
n_class=39
param['num_class'] = n_class
num_round = 30

watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_valid );

yprob = bst.predict( xg_valid ).reshape( valid_Y.shape[0], n_class )
ylabel = np.argmax(yprob, axis=1)
error=evalerror(ylabel,valid_Y)



yprob = bst.predict( xg_test ).reshape( test_X.shape[0], n_class )
#output test set probabilities to csv file
columns=['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY',
            'BURGLARY', 'DISORDERLY CONDUCT',
            'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC',
            'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
            'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD',
            'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT',
            'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
            'NON-CRIMINAL', 'OTHER OFFENSES',
            'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION',
            'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
            'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
            'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
            'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS',
            'VANDALISM', 'VEHICLE THEFT', 'WARRANTS',
            'WEAPON LAWS']
df = pd.DataFrame(columns=['Id']+columns)
df=pd.DataFrame(yprob,columns=columns)
df.insert(loc=0,column='Id',value=range(len(df)))
# np.savetxt("RF_predictions.csv.gz", df, delimiter=",")
df.to_csv("XGB_predictions.csv",sep=",",index=False)
####################################################################################
####################################################################################
####################################################################################
# bag estimates from non-overlapping folds
####################################################################################
####################################################################################
####################################################################################
p=range(50)
frac_=0.02
r0=range(sz[0])
X_folds=np.zeros(len(p)-1)
X_folds=np.zeros(len(p)-1)
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
      yprob = bst.predict( xg_valid ).reshape( valid_Y.shape[0], n_class )
      ylabel = np.argmax(yprob, axis=1)
      error=evalerror(ylabel,valid_Y)
      X_folds[i]=error
      y_test = bst.predict( xg_test );
      y_test_mat[:,i]=y_test
#bag estimates from the model trained on different folds (no need to average since ranking only matter)
y_bag=y_test_mat.sum(axis=1)

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
child_grid=[1,3,5]
n_child=len(child_grid)
eta_grid=[0.1]
n_eta=len(eta_grid)
gamma_grid=[0.0,0.1,0.4]
n_gamma=len(gamma_grid)
subsample_grid=[0.7,0.75,0.80]
n_subsample=len(subsample_grid)

X_cv=np.zeros((n_child*n_eta*n_gamma*n_subsample,5))
ind=0
for i in range(n_child):
      for j in range(n_eta):
            for k in range(n_gamma):
                  for l in range(n_subsample):
                        param = {}
                        param['min_child_weight']=child_grid[i]
                        param['eta']=eta_grid[j]
                        param['alpha']=0.1
                        param['gamma']=gamma_grid[k]
                        param["subsample"] = subsample_grid[l]
                        param['objective'] = 'multi:softprob'
                        # params["scale_pos_weight"] = 1.0
                        param['silent'] = 1
                        param['nthread'] = 4
                        n_class=39
                        param['num_class'] = n_class

                        watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
                        num_round = 5

                        bst = xgb.train(param, xg_train, num_round, watchlist );
                        yprob = bst.predict( xg_valid ).reshape( valid_Y.shape[0], n_class )
                        ylabel = np.argmax(yprob, axis=1)
                        error=evalerror(ylabel,valid_Y)
                        X_cv[ind,0]=child_grid[i]
                        X_cv[ind,1]=eta_grid[j]
                        X_cv[ind,2]=gamma_grid[k]
                        X_cv[ind,3]=subsample_grid[l]
                        X_cv[ind,4]=error
                        ind+=1
                        print ind

df_cv=pd.DataFrame(X_cv)
df_cv.to_csv('xgboost_cv_v2.csv')

####################################################################################
####################################################################################
####################################################################################
