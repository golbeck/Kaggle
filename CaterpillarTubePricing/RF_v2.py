
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

from sklearn.ensemble import RandomForestRegressor

####################################################################################
####################################################################################
####################################################################################


X_train=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
col_names=X_train.columns
Y_train=np.array(pd.io.parsers.read_table('Y_train.csv',sep=',',header=False))
X_train['Ind']=range(X_train.shape[0])
X_test=pd.io.parsers.read_table('X_test.csv',sep=',',header=False)
X_test['Ind']=range(X_test.shape[0])
indices=[i+1 for i in range(X_test.shape[0])]


#first train model on products with bracket pricing
X_dat=X_train[X_train['bracket_pricing']==0.0].copy()
train_indices=X_dat['Ind']
X_dat=X_dat.drop('Ind', axis=1)
X_dat=X_dat.drop('bracket_pricing', axis=1)
X_dat=np.array(X_dat)
Y_dat=Y_train[list(train_indices),:]
Y_dat=np.log1p(Y_dat)

test_X=X_test[X_test['bracket_pricing']==0.0]
bracket_0_indices=test_X['Ind']
test_X=test_X.drop('Ind', axis=1)
test_X=test_X.drop('bracket_pricing', axis=1)
test_X=np.array(test_X)
####################################################################################
####################################################################################
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
#classifier
RFmodel = RandomForestRegressor(
        n_estimators=1000,        #number of trees to generate
        n_jobs=1,               #run in parallel on all cores
        criterion="mse"
        )

frac=0.95
train_X = X_dat[:int(sz[0] * frac), :]
train_Y = Y_dat[:int(sz[0] * frac)]
train_Y=train_Y.reshape(len(train_Y,))
valid_X = X_dat[int(sz[0] * frac):, :]
valid_Y = Y_dat[int(sz[0] * frac):]
valid_Y=valid_Y.reshape(len(valid_Y,))

#train
RFmodel = RFmodel.fit(train_X, train_Y)
#get parameters
params=RFmodel.get_params()
#score on training set
acc_rate=RFmodel.score(train_X,train_Y)
print acc_rate
#feature importances
feat_imp=RFmodel.feature_importances_
df_train=pd.io.parsers.read_table('X_train.csv',sep=',',header=False)
col_names=list(df_train.columns)
feat_imp_dict={col_names[i]:feat_imp[i] for i in range(len(feat_imp))}
feat_imp_sort = sorted(feat_imp_dict.items(), key=operator.itemgetter(1))


pred=RFmodel.predict(valid_X)
print ('prediction error=%f' % np.sqrt(sum( (pred[i]-valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))

#predict probabilities
y_test=RFmodel.predict(X_test)
df=pd.DataFrame(np.expm1(y_test))
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("RF_predictions.csv",sep=",",index=False)