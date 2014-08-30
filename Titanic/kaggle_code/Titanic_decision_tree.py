#initiaize with ipython -pylab
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


df = pd.read_csv("data/train.csv")
df = df.drop(['ticket','cabin'], axis=1) 
df = df.dropna()
