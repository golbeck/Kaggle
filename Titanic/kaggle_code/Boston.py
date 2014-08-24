#This data frame contains the following columns:

#     ‘crim’ per capita crime rate by town.

#     ‘zn’ proportion of residential land zoned for lots over 25,000
#          sq.ft.

#     ‘indus’ proportion of non-retail business acres per town.

#     ‘chas’ Charles River dummy variable (= 1 if tract bounds river; 0
#          otherwise).

#     ‘nox’ nitrogen oxides concentration (parts per 10 million).

#     ‘rm’ average number of rooms per dwelling.

#     ‘age’ proportion of owner-occupied units built prior to 1940.
#     
#     ‘dis’ weighted mean of distances to five Boston employment
#          centres.

#     ‘rad’ index of accessibility to radial highways.

#     ‘tax’ full-value property-tax rate per \$10,000.

#     ‘ptratio’ pupil-teacher ratio by town.

#     ‘black’ 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by
#          town.

#     ‘lstat’ lower status of the population (percent).

#     ‘medv’ median value of owner-occupied homes in \$1000s.


# http://scikit-learn.org/stable/modules/generated/sklearn.lda.LDA.html#sklearn.lda.LDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.lda import LDA


Boston=pd.read_csv("boston.csv")
my_cols=set(Boston.columns)
my_cols.remove('Unnamed: 0')
my_cols.remove('medv')
my_cols=list(my_cols)
X=Boston[my_cols]
X=np.array(X)
Y=np.array(Boston['medv'])

Boston_lda=LDA()
Boston_lda.fit(X,Y)

X_test=pd.DataFrame(Y)
X_test['predict']=pd.Series(Boston_lda.predict(X),index=X_test.index)
X_test.columns=['data','predict']
plt.scatter(X_test['data'],X_test['predict'])


