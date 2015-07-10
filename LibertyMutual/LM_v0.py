import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from sklearn.linear_model import LinearRegression
%pylab
import matplotlib.pyplot as plt
####################################################################################
####################################################################################
####################################################################################
pwd_temp=os.getcwd()
dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
# dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

X_train=np.loadtxt("X_train.gz",delimiter=",")
Y_train=np.loadtxt("Y_train.gz",delimiter=",")
n=X_train.shape[0]

#shuffle train set
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(Y_train)        

X_test=np.loadtxt("X_test.gz",delimiter=",")

frac=0.95
n_train_set=int(frac*n)
train_set_x=X_train[range(n_train_set),:]
train_set_y=Y_train[range(n_train_set),]

valid_set_x=X_train[range(n_train_set,n),:]
valid_set_y=Y_train[range(n_train_set,n),]
####################################################################################
####################################################################################
####################################################################################
#model
LRmodel=LinearRegression(
    fit_intercept=True, 
    normalize=False, 
    copy_X=True
    )

#train
LRmodel = LRmodel.fit(train_set_x, train_set_y)

var_score=np.mean((LRmodel.predict(valid_set_x) - valid_set_y) ** 2)
# The coefficients
print('Coefficients: \n', LRmodel.coef_)
# The mean square error
print("Residual sum of squares: %.2f"  % var_score)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % LRmodel.score(valid_set_x, valid_set_y))
####################################################################################
####################################################################################
####################################################################################
#predict 
y_test=LRmodel.predict(X_test)
print y_test
df=pd.DataFrame(y_test)
df.columns=['Hazard']
indices=np.loadtxt("X_test_indices.gz",delimiter=",").astype('int32')
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("LR_predictions.csv",sep=",",index=False)