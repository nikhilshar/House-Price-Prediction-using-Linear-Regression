
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd

boston=load_boston()
df_x= pd.DataFrame(boston.data, columns= boston.feature_names)
df_y=pd.DataFrame(boston.target)

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state =4)
x_train.describe()
reg=LinearRegression()
reg.fit(x_train,y_train)
a= reg.predict(x_test)
print(a)

p=(a-y_test)**2
print(p.mean())