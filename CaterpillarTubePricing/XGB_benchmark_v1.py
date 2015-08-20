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


# label encode the categorical variables
for i in range(train.shape[1]):
    if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34]:
        print(i,list(train.ix[1:5,i]) + list(test.ix[1:5,i]))
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train.ix[:,i]) + list(test.ix[:,i]))
        train.ix[:,i] = lbl.transform(train.ix[:,i])
        test.ix[:,i] = lbl.transform(test.ix[:,i])


# convert data to numpy array
train = np.array(train)
test = np.array(test)
# object array to float
train = train.astype(float)
test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;) 
# The choice is yours :)
label_log = np.log1p(labels)

# fit a random forest model

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.02
params["min_child_weight"] = 6
params["subsample"] = 0.7
params["colsample_bytree"] = 0.6
params["scale_pos_weight"] = 0.8
params["silent"] = 1
params["max_depth"] = 8
params["max_delta_step"]=2

plst = list(params.items())


sz = train.shape
xg_train = xgb.DMatrix( train, label=label_log)
xgtest = xgb.DMatrix(test)
watchlist = [ (xg_train,'train') ]

print('2000')


num_round = 2000
bst = xgb.train(params, xg_train, num_round);
preds1 = bst.predict(xgtest)

print('3000')

num_round = 3000


bst = xgb.train(params, xg_train, num_round);
preds2 = bst.predict(xgtest)

print('4000')

num_round = 4000
bst = xgb.train(params, xg_train, num_round);
preds4 = bst.predict(xgtest)

label_log = np.power(labels,1/16)

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)

print('power 1/16 4000')

num_round = 4000

watchlist = [ (xg_train,'train') ]
bst = xgb.train(params, xg_train, num_round);
preds3 = bst.predict(xgtest)

#for loop in range(2):
#    model = xgb.train(plst, xgtrain, num_rounds)
#    preds1 = preds1 + model.predict(xgtest)
preds = 0.4*np.expm1(preds4)+.1*np.expm1(preds1)+0.1*np.expm1(preds2)+0.4*np.power(preds3,16)
#preds = (0.58*np.expm1( (preds1+preds2+preds4)/3))+(0.42*np.power(preds3,16))

preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('benchmark.csv', index=False)
