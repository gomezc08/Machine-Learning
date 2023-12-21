import numpy as np
import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

x = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]
y2 = df_test.iloc[:, 0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# MULTIPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(df_test) 
print(y_pred.shape)