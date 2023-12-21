import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("salaries")
print(df)

"""
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# preprocessing polynomial linear regression.
from sklearn.preprocessing import PolynomialFeatures
obj = PolynomialFeatures(degree = 3)
xPoly = obj.fit_transform(x)

# avoiding dummy variable trap 
x = x[:, 1:]

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# predict test results
y_pred = lr.predict(x_test) 

from sklearn import metrics
# you can compare outputted values of y_pred to y_test to see how close our prediction is to the actual value...
print(metrics.mean_squared_error(y_test, y_pred))   # error = high b/c of low number of data / test runs.
"""