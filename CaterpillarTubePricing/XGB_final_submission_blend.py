import numpy as np
import pandas as pd
X1=(pd.io.parsers.read_table('XGB_predictions.csv',sep=',',header=False))
X2=(pd.io.parsers.read_table('XGB_predictions1.csv',sep=',',header=False))
X3=(pd.io.parsers.read_table('XGB_predictions2.csv',sep=',',header=False))

X=pd.merge(X1,X2,on='Id')
X=pd.merge(X,X3,on='Id')

df_blend=pd.DataFrame()
df_blend['Id']=X['Id']
df_blend['cost']=X.ix[:,1:].mean(1)

df_blend.to_csv("XGB_predictions_blend.csv",sep=",",index=False)