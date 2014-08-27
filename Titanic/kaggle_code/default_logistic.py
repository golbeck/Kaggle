# http://scikit-learn.org/stable/modules/generated/sklearn.lda.LDA.html#sklearn.lda.LDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.utils.extmath import cartesian
from sklearn.metrics import confusion_matrix
import pylab as pl

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
 
# fit the model
logit=sm.Logit(Ydf['default_Yes'],Xdf)
result=logit.fit()
print result.summary()
#odds ratio components
print np.exp(result.params)
#model parameters
beta=pd.DataFrame(result.params)


#classify if prediction > 0.5
Y_out=pd.DataFrame()
Y_out['df']=result.predict(Xdf)
Y_out.df[Y_out['df']>0.5]=1.0
Y_out.df[Y_out['df']<=0.5]=0.0

#confusion matrix
cm = confusion_matrix(Ydf['default_Yes'],Y_out['df'])
print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#for the plots, create arrays of outputs of the continuous predictor variables
h=50.0
balance_=np.linspace(Xdf['balance'].min(),Xdf['balance'].max(),h)
h=100.0
income_=np.linspace(Xdf['income'].min(),Xdf['income'].max(),h)
#create combinations of the predictors using the arrays above
combos = pd.DataFrame(cartesian([balance_,[0.0,1.0],income_,[1.]]))
combos.columns=['balance','student','income','intercept']
#run the fitted model on all the predictor combinations to obtain predicted probabilities of default
combos['predict']=result.predict(combos)

#return the predicted probability of default for the mean income level, 
#and for each level of balance and student status
grouped = pd.pivot_table(combos,values=['predict'],rows=['balance','student'],aggfunc=np.mean)
#select only data with 'student'=1
plt.figure()
plt_data=grouped.ix[grouped.index.get_level_values(1)==1]
#plot predicted probability of default for 'student'=1
plt.plot(plt_data.index.get_level_values(0),plt_data['predict'],color='b')
#select only data with 'student'=0
plt_data=grouped.ix[grouped.index.get_level_values(1)==0]
#plot predicted probability of default for 'student'=0
plt.plot(plt_data.index.get_level_values(0),plt_data['predict'],color='g')
#annotate plot
plt.xlabel('balance')
plt.ylabel("P(default)")
plt.legend(['1','0'], loc='upper left',title='student')
plt.title("Prob(default) isolating balance and student status")
plt.show()


#plot the decision boundary for 'student'=1
#create a dataframe of 'student' status and predicted default probabilities
Xstudent=pd.DataFrame(Xdf['student'])
Xstudent['df']=Y_out['df']
#only use the predicted defaults if 'student'=1
categories = np.array(Xstudent[Xstudent['student']==1.0]['df']).astype('int64')
colormap = np.array(['r', 'g'])
plt.figure()
plt.scatter(Xdf[Xdf['student']==1.0]['balance'], Xdf[Xdf['student']==1.0]['income'], s=50, c=colormap[categories])
#generate the points (balance,income) that corresponds to a zero log odds ratio. This is the decision boundary
income_=-(beta.ix['intercept',0]+beta.ix['student',0]+beta.ix['balance',0]*balance_)/beta.ix['income',0]
#plot the decision boundary
plt.plot(balance_,income_)
#axis labels
plt.xlabel('balance')
plt.ylabel('income')
#set y and x axis boundaries
plt.ylim((Xdf['income'].min(),Xdf['income'].max()))
plt.xlim((Xdf['balance'].min(),Xdf['balance'].max()))
plt.title("Classification for 'student'=1")
plt.show()


#plot the decision boundary for 'student'=0
#create a dataframe of 'student' status and predicted default probabilities
Xstudent=pd.DataFrame(Xdf['student'])
Xstudent['df']=Y_out['df']
#only use the predicted defaults if 'student'=0
categories = np.array(Xstudent[Xstudent['student']==0.0]['df']).astype('int64')
colormap = np.array(['r', 'g'])
plt.figure()
plt.scatter(Xdf[Xdf['student']==0.0]['balance'], Xdf[Xdf['student']==0.0]['income'], s=50, c=colormap[categories])
#generate the points (balance,income) that corresponds to a zero log odds ratio. This is the decision boundary
income_=-(beta.ix['intercept',0]+beta.ix['balance',0]*balance_)/beta.ix['income',0]
#plot the decision boundary
plt.plot(balance_,income_)
#axis labels
plt.xlabel('balance')
plt.ylabel('income')
#set y and x axis boundaries
plt.ylim((Xdf['income'].min(),Xdf['income'].max()))
plt.xlim((Xdf['balance'].min(),Xdf['balance'].max()))
plt.title("Classification for 'student'=0")
plt.show()


#return the predicted probability of default for the mean balance level, 
#and for each level of income and student status
grouped = pd.pivot_table(combos,values=['predict'],rows=['income','student'],aggfunc=np.mean)
#select only data with 'student'=1
plt.figure()
plt_data=grouped.ix[grouped.index.get_level_values(1)==1]
#plot predicted probability of default for 'student'=1
plt.plot(plt_data.index.get_level_values(0),plt_data['predict'],color='b')
#select only data with 'student'=0
plt_data=grouped.ix[grouped.index.get_level_values(1)==0]
#plot predicted probability of default for 'student'=0
plt.plot(plt_data.index.get_level_values(0),plt_data['predict'],color='g')
#annotate plot
plt.xlabel('balance')
plt.ylabel("P(default)")
plt.legend(['1','0'], loc='upper left',title='student')
plt.title("Prob(default) isolating income and student status")
plt.show()
