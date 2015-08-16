#forked from Gilberto Titericz Junior


import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import xgboost as xgb

# load training and test datasets
train = pd.read_csv('train_set.csv', parse_dates=[2,])
test = pd.read_csv('test_set.csv', parse_dates=[3,])
tube_data = pd.read_csv('tube.csv')
bill_of_materials_data = pd.read_csv('bill_of_materials.csv')
specs_data = pd.read_csv('specs.csv')

print("train columns")
print(train.columns)
print("test columns")
print(test.columns)
print("tube.csv df columns")
print(tube_data.columns)
print("bill_of_materials.csv df columns")
print(bill_of_materials_data.columns)
print("specs.csv df columns")
print(specs_data.columns)

print(specs_data[2:3])

train = pd.merge(train, tube_data, on ='tube_assembly_id')
train = pd.merge(train, bill_of_materials_data, on ='tube_assembly_id')
test = pd.merge(test, tube_data, on ='tube_assembly_id')
test = pd.merge(test, bill_of_materials_data, on ='tube_assembly_id')

print("new train columns")
print(train.columns)
print(train[1:10])
print(train.columns.to_series().groupby(train.dtypes).groups)

# create some new features
train['year'] = pd.to_datetime(train.quote_date)
train['year'] = np.array([x.year for x in train['year']])
train['month'] = pd.to_datetime(train.quote_date)
train['month'] = np.array([x.month for x in train['month']])

test['year'] = pd.to_datetime(test.quote_date)
test['year'] = np.array([x.year for x in test['year']])
test['month'] = pd.to_datetime(test.quote_date)
test['month'] = np.array([x.month for x in test['month']])

# drop useless columns and create labels
idx = test.id.values.astype(int)
test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
labels = train.cost.values
#'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x'
#for some reason material_id cannot be converted to categorical variable
train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

train['material_id'].replace(np.nan,' ', regex=True, inplace= True)
test['material_id'].replace(np.nan,' ', regex=True, inplace= True)
for i in range(1,9):
    column_label = 'component_id_'+str(i)
    print(column_label)
    train[column_label].replace(np.nan,' ', regex=True, inplace= True)
    test[column_label].replace(np.nan,' ', regex=True, inplace= True)

train.fillna(0, inplace = True)
test.fillna(0, inplace = True)

print("train columns")
print(train.columns)

# convert data to numpy array
train = np.array(train)
test = np.array(test)


# label encode the categorical variables
for i in range(train.shape[1]):
    if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34]:
        print(i,list(train[1:5,i]) + list(test[1:5,i]))
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])


# object array to float
train = train.astype(float)
test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;) 
# The choice is yours :)
label_log = np.log1p(labels)

# fit a random forest model

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 6
params["subsample"] = 0.87
params["colsample_bytree"] = 0.50
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 7


plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

num_rounds = 7000
model = xgb.train(plst, xgtrain, num_rounds)
y_out = model.predict( xgtrain );
print ('prediction error=%f' % np.sqrt(sum( (y_out[i]-label_log[i])**2 for i in range(len(label_log))) / float(len(label_log)) ))


preds1 = model.predict(xgtest)
preds = np.expm1(preds1)

preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('yx_benchmark--LB0.2369.csv', index=False)


####################################################################################
####################################################################################
####################################################################################
sz=train.shape
frac=0.95
train_X = train[:int(sz[0] * frac), :]
train_Y = label_log[:int(sz[0] * frac)]
valid_X = train[int(sz[0] * frac):, :]
valid_Y = label_log[int(sz[0] * frac):]

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_valid = xgb.DMatrix(valid_X, label=valid_Y)
xg_test = xgb.DMatrix(test)
# setup parameters for xgboost

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.1
params["min_child_weight"] = 6
params["subsample"] = 0.87
params["colsample_bytree"] = 0.50
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 7
num_round = 7000

watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
bst = xgb.train(params, xg_train, num_round, watchlist ,early_stopping_rounds=50);
n_tree = bst.best_iteration
pred = bst.predict( xg_valid,ntree_limit=n_tree );
print ('prediction error=%f' % np.sqrt(sum( (pred[i]-valid_Y[i])**2 for i in range(len(valid_Y))) / float(len(valid_Y)) ))
