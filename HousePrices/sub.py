import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

x = df_train.iloc[:, [1, 4, 7, 10, 12, 16, 17, 18, 19, 20, 21, 38, 46, 49, 50, 54, 62]]
y = df_train.iloc[:, -1]
x_values = x.values
y = y.values


def drop():
    global x_values
    

def le(num):
    global x_values
    from sklearn.preprocessing import LabelEncoder
    l = LabelEncoder()
    x_values[:, num] = l.fit_transform(x_values[:, num])
    
def ohe(name, num):
    global x_values
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    ct = ColumnTransformer([(name, OneHotEncoder(), [num])], remainder = 'passthrough')
    x_values = ct.fit_transform(x_values)

categorical_features = x.select_dtypes(include=['object']).columns

# start
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
x_values[:, 2] = labelEncoder_X.fit_transform(x_values[:, 2])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("LotShape", OneHotEncoder(), [2])], remainder = 'passthrough')
x_values = ct.fit_transform(x_values)
x_values = x_values[:, 1:]
# end

# start
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
x_values[:, 5] = labelEncoder_X.fit_transform(x_values[:, 5])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("LotConfig", OneHotEncoder(), [5])], remainder = 'passthrough')
x_values = ct.fit_transform(x_values)
x_values = x_values[:, 1:]
# end

# start
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
x_values[:, 9] = labelEncoder_X.fit_transform(x_values[:, 9])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Neighborhood", OneHotEncoder(), [9])], remainder = 'passthrough')
x_values = ct.fit_transform(x_values)
x_values = x_values[:, 1:]
# end

# start
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
x_values[:, 33] = labelEncoder_X.fit_transform(x_values[:,33])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("HouseStyle", OneHotEncoder(), [33])], remainder = 'passthrough')
x_values = ct.fit_transform(x_values)
x_values = x_values[:, 1:]
# end


# start
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
x_values[:, 44] = labelEncoder_X.fit_transform(x_values[:,44])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("RoofStyle", OneHotEncoder(), [44])], remainder = 'passthrough')
x_values = ct.fit_transform(x_values)
x_values = x_values[:, 1:]
# end


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(x_values)


"""6. Dataset Splition"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_values,y,test_size= 1/3, random_state = 0)

"""7. Modeling"""

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

"""8. Predictions"""

# Predicting the Test set results
y_pred = linear_regressor.predict(X_test)

"""9. Performance Evaluation
"""

# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the results using MAE
print(metrics.mean_absolute_error(y_test, y_pred))

# Evaluating the model and printing the results using MSE
print(metrics.mean_squared_error(y_test, y_pred))

# Importing metrics library
from sklearn import metrics
# Evaluating the model and printing the results using MAE
final = metrics.mean_absolute_error(y_test, y_pred)

# Evaluating the model and printing the results using MSE
print(metrics.mean_squared_error(y_test, y_pred))


























"""5. Avoiding the Dummy Variable Trap"""

# Avoiding the Dummy Variable Trap


#for i in categorical_features:
 #   df = pd.DataFrame(x_values)
  #  print(df[0].columns)
    #le(x_values)
    #ohe(i, x_values.get_loc(i))
    #x_values = x_values[:, 1:]
    #x = pd.DataFrame(x_values)
"""
for i in range(0, x_values.shape[1]):
    if isinstance(x.iloc[1, i], str) == True:
        # wanna add it to cat
        le(i)
        ohe(x.columns[i], i)
        x_values = x_values[:, 1:]
"""
#from sklearn.model_selection import train_test_split
#(x_train, x_test, y_train, y_test) = train_test_split(x_values, y_values, test_size = 1/5, random_state=0)
