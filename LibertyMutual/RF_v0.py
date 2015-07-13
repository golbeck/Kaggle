import os
import sys
import time
import datetime

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
dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
# dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

X_train=np.loadtxt("X_train_pca_v2.gz",delimiter=",")
X_test=np.loadtxt("X_test_pca_v2.gz",delimiter=",")
Y_train=np.loadtxt("Y_train.gz",delimiter=",")  
n=X_train.shape[0]

#shuffle train set
rng_state = np.random.get_state()
#randomly permuate the features and outputs using the same shuffle for each epoch
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(Y_train)
####################################################################################
####################################################################################
####################################################################################
#classifier
RFmodel = RandomForestRegressor(
        n_estimators=100,        #number of trees to generate
        n_jobs=1,               #run in parallel on all cores
        criterion="mse"
        )

#train
RFmodel = RFmodel.fit(X_train, Y_train)
#get parameters
params=RFmodel.get_params()
#score on training set
acc_rate=RFmodel.score(X_train,Y_train)
print acc_rate
#feature importances
feat_imp=RFmodel.feature_importances_
#predict probabilities
y_test=RFmodel.predict(X_test)

print y_test
df=pd.DataFrame(y_test)
df.columns=['Hazard']
indices=np.loadtxt("X_test_indices.gz",delimiter=",").astype('int32')
df.insert(loc=0,column='Id',value=indices)
# np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
df.to_csv("RF_pca_predictions.csv",sep=",",index=False)