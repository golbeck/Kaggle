import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from sklearn.ensemble import RandomForestClassifier

####################################################################################
####################################################################################
####################################################################################
pwd_temp=os.getcwd()
# dir1='/home/sgolbeck/workspace/Kaggle/SFcrime'
dir1='/home/golbeck/Workspace/Kaggle/SFcrime'
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
####################################################################################
####################################################################################
####################################################################################
#classifier
RFmodel = RandomForestClassifier(
        n_estimators=40,        #number of trees to generate
        max_features='auto',    #consider sqrt of number of features when splitting
        n_jobs=1               #run in parallel on all cores
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
test_probs=RFmodel.predict_proba(X_test)

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
df=pd.DataFrame(test_probs,columns=columns)
df.insert(loc=0,column='Id',value=range(len(df)))
# np.savetxt("RF_predictions.csv.gz", df, delimiter=",")
df.to_csv("RF_predictions.csv",sep=",",index=False)