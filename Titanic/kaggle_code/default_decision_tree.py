# http://scikit-learn.org/stable/modules/tree.html
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import pylab as pl
import pydot 
import os
from os import system
import random
from sklearn.metrics import confusion_matrix

#download default data and create a matrix of predictors
Default=pd.read_csv("Default.csv")
my_cols=set(Default.columns)
my_cols.remove('Unnamed: 0')
my_cols.remove('default')
my_cols=list(my_cols)
X=Default[my_cols]
X=np.array(X)

#convert matrix of predictors to a dataframe
Xdf=pd.DataFrame(X)
Xdf.columns=['balance','student','income']
#create dummy variables from the 'student' column
dummy_ranks=pd.get_dummies(Xdf['student'],prefix='student')
#removed 'student' column
Xdf.drop('student',1)
#create new 'student' column using one of the resulting dummy variable outputs
Xdf['student']=dummy_ranks['student_Yes']
#create a column for the intercept in the logit model
Xdf['intercept']=1.0
#convert all data to floats
Xdf=Xdf.astype('float')
#default indicator
Ydf=pd.get_dummies(Default['default'],prefix='default').astype('float')

#training data
rows = random.sample(Xdf.index, 9000)
Xdf_train=Xdf.ix[rows]
Ydf_train=Ydf.ix[rows]
#test data
Xdf_test=Xdf.drop(rows)
Ydf_test=Ydf.drop(rows)

#decision tree classification
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xdf_train, Ydf_train['default_Yes'])
clf_pred=pd.DataFrame(clf.predict(Xdf_test))

print pd.DataFrame(clf.feature_importances_, columns = ["Imp"], index = Xdf.columns).sort(['Imp'], ascending = False)


#confusion matrix
cm = confusion_matrix(Ydf_test['default_Yes'],clf_pred)
print(cm)

plt.figure()
# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#plot decision tree
dotfile = open("clf.dot", 'w')
dotfile = tree.export_graphviz(clf, out_file = dotfile, feature_names = Xdf.columns)
system("dot -Tpng clf.dot -o default.png")
os.unlink('clf.dot')

