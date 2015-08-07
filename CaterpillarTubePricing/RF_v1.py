
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
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/CaterpillarTubePricing'
# dir1='/home/golbeck/Workspace/Kaggle/CaterpillarTubePricing'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

X_train=np.array(pd.io.parsers.read_table('X_train.csv',sep=',',header=False))
Y_train=np.array(pd.io.parsers.read_table('Y_train.csv',sep=',',header=False))
Y_train=Y_train.reshape(len(Y_train,))
X_test=np.array(pd.io.parsers.read_table('X_test.csv',sep=',',header=False))
n=X_train.shape[0]
indices=[i+1 for i in range(X_test.shape[0])]


#shuffle train set
seed=1234
np.random.seed(seed=seed)
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(Y_train)

sz = X_train.shape

frac=0.8
train_X = X_train[:int(sz[0] * frac), :]
train_Y = Y_train[:int(sz[0] * frac)]
valid_X = X_train[int(sz[0] * frac):, :]
valid_Y = Y_train[int(sz[0] * frac):]
####################################################################################
####################################################################################
####################################################################################
#classifier
RFmodel = RandomForestRegressor(
        n_estimators=1000,        #number of trees to generate
        n_jobs=1,               #run in parallel on all cores
        criterion="mse"
        )

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


y_out=RFmodel.predict(valid_X)
pred = np.array([np.max([0.0,x]) for x in y_out])
print ('prediction error=%f' % (sum( (np.log(pred[i]+1.0) - np.log(valid_Y[i]+1.0))**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))

#predict probabilities
y_out=RFmodel.predict(X_test)
y_test = np.array([np.max([0.0,x]) for x in y_out])
df=pd.DataFrame(y_test)
df.columns=['cost']
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("RF_predictions.csv",sep=",",index=False)