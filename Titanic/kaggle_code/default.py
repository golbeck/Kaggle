# http://scikit-learn.org/stable/modules/generated/sklearn.lda.LDA.html#sklearn.lda.LDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

Default=pd.read_csv("Default.csv")
my_cols=set(Default.columns)
my_cols.remove('Unnamed: 0')
my_cols.remove('default')
my_cols=list(my_cols)
X=Default[my_cols]
X=np.array(X)

#convert from text to numeric
enc=LabelEncoder()
label_encoder=enc.fit(X[:,1])
integer_classes=label_encoder.transform(label_encoder.classes_)
t=label_encoder.transform(X[:,1])
X[:,1]=t

#response variable
Y=np.array(Default['default'])
#convert from text to numeric
label_encoder=enc.fit(Y)
integer_classes=label_encoder.transform(label_encoder.classes_)
t=label_encoder.transform(Y)
Y=t

#all data converted to float type
X=X.astype('float')
Y=Y.astype('float')

#perform linear discriminant analysis
Default_lda=LDA()
Default_lda.fit(X,Y)

X_test=pd.DataFrame(Y)
X_test['predict']=pd.Series(Default_lda.predict(X),index=X_test.index)
X_test.columns=['data','predict']
#plt.scatter(X_test['data'],X_test['predict'])

#confusion matrix
cm = confusion_matrix(X_test['data'],X_test['predict'])
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#plot predictions with continuous predictors
X_out=np.array(X[:,[0,2]])
categories = np.array(X_test['predict']).astype('int64')
colormap = np.array(['r', 'g'])
#plt.scatter(X_out[:,0], X_out[:,1], s=50, c=colormap[categories])

#plot actual data with continuous predictors
Y1=pd.DataFrame(Y)
Y1.columns=['data']
Y1['marker']='*'
for i in range(len(Y)):
    if Y1.ix[i,'data']==0.0:
        Y1.ix[i,'marker']='x'

markers=list(set(Y1.marker))
Y2=pd.DataFrame(markers)
Y2.columns=['marker']
Y2['colors']=['r','g']

f, axarr = plt.subplots(2)
axarr[0].scatter(X_out[:,0], X_out[:,1], s=50, c=colormap[categories])
for m in np.unique(Y1.marker):
    selector = Y1.marker == m
    axarr[1].scatter(Default[selector].balance, Default[selector].income, c=Y1[selector].data,marker=m,color=Y2[Y2.marker==m].colors)
