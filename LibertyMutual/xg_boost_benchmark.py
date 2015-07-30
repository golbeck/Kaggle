import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer



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


def xgboost_pred(train,labels,test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["min_child_weight"] = 25
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.85
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 10

    plst = list(params.items())

    #Using 8000 rows for early stopping. 
    offset = 8000

    num_rounds = 5000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices 
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
#   xgtrain = xgb.DMatrix(train, label=labels)
    #xgval = xgb.DMatrix(train, label=labels)

    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    preds_valid = model.predict(xgval)
    valid_gini=Gini(np.array(labels[:offset,]), preds_valid[:,])
    print valid_gini
    #model = xgb.train(plst, xgtrain, 1000)
    preds1 = model.predict(xgtest)


    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
    train = train[::-1,:]
    labels = np.log(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])
#   xgtrain = xgb.DMatrix(train, label=labels)


    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
    preds_valid = model.predict(xgval)
    valid_gini=Gini(np.array(labels[:offset,]), preds_valid[:,])
    print valid_gini
#   model = xgb.train(plst, xgtrain, 1000)
    preds2 = model.predict(xgtest)


    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    preds = preds1*0.4 + preds2*0.6
    return preds

#load train and test 
train  = pd.read_csv('train.csv', index_col=0)
test  = pd.read_csv('test.csv', index_col=0)


labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

train_s = train
test_s = test


train_s.drop('T2_V10', axis=1, inplace=True)
train_s.drop('T2_V7', axis=1, inplace=True)
train_s.drop('T1_V13', axis=1, inplace=True)
train_s.drop('T1_V10', axis=1, inplace=True)

test_s.drop('T2_V10', axis=1, inplace=True)
test_s.drop('T2_V7', axis=1, inplace=True)
test_s.drop('T1_V13', axis=1, inplace=True)
test_s.drop('T1_V10', axis=1, inplace=True)

columns = train.columns
test_ind = test.index


train_s = np.array(train_s)
test_s = np.array(test_s)

# label encode the categorical variables
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)


preds1 = xgboost_pred(train_s,labels,test_s)

#model_2 building

train = train.T.to_dict().values()
test = test.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

preds2 = xgboost_pred(train,labels,test)


preds = 0.5 * preds1 + 0.5 * preds2

# #generate solution
# preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
# preds = preds.set_index('Id')
# preds.to_csv('xgboost_benchmark_ds.csv')
                