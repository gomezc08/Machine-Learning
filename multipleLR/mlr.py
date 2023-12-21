import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""2. Data Acquistion"""
dataset = pd.read_csv('50_AdAgency.csv')

"""3.  Creating Data Frames
"""

# Creating Data Frames
X = dataset.iloc[:, :- 1].values
Y = dataset.iloc[:, 4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Geography", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)



# Avoiding the Dummy Variable Trap
X = X[:, 1:]

print(X.shape)
print(Y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size= 1/3, random_state = 0)
#print(X_train.shape)
#print(y_train.shape)


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = linear_regressor.predict(X_test)



# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the results using MAE
print(metrics.mean_absolute_error(y_test, y_pred))

# Evaluating the model and printing the results using MSE
print(metrics.mean_squared_error(y_test, y_pred))

# Importing the math library
import math

# Evaluating the model and printing the results using RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

