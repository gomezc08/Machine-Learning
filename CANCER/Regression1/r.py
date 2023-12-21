import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in data 
df = pd.read_csv("data.csv")
x1 = df.iloc[:, 0:1]
y1 = df.iloc[:, -1]

from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=0)

# Building model / training...
from sklearn.linear_model import LinearRegression
lRegressor = LinearRegression() 
lRegressor.fit(x1_train, y1_train)

# Predicting the test set results...
y1_pred = lRegressor.predict(x1_test)

from sklearn import metrics
# you can compare outputted values of y_pred to y_test to see how close our prediction is to the actual value...
print(metrics.mean_squared_error(y_test, y_pred))   # error = high b/c of low number of data / test runs.

# visualizing data...
plt.scatter(x1_train, y1_train, color="red")
plt.plot(x1_train, lRegressor.predict(x1_train), color = "blue")
plt.title("Visualization of Training Data")
plt.xlabel("Years of Research Expereince")
plt.ylabel("Stipend")
plt.show()
