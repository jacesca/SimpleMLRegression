from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score)
import statsmodels.api as sm   # import statsmodels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# Reading the data
print("""
-----------------
Reading the data:
-----------------
""")
df = pd.read_csv('data/simple_height_data.csv')
print('(1). Get an overview of the data')
print(df.head(10))

print('\n\n\n(2). Check missing values: ')
print(df.info())

# Preprocessing data with scikit-learn
print("""\n\n\n
-----------------
Preprocessing the data:
-----------------
""")
print('(1). Separate the data into x and y')
x, y = df['Father'], df['Height']

print('\n\n\n(2). Separate the data into training and testing set')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Modeling
print("""\n\n\n
-----------------
Modeling:
-----------------
""")
print('(1). Get the correct form of input for OLS')
x_train_tilde = sm.add_constant(x_train)

print('\n\n\n(2). Initialize an OLS object')
regression_model = sm.OLS(y_train, x_train_tilde)

print('\n\n\n(3). Train the model')
regression_model = regression_model.fit()

print('\n\n\n(4). Get the parameters')
beta_0, beta_1 = regression_model.params
print('beta_0 is', beta_0)
print('beta_1 is', beta_1)

print('\n\n\n(5). Print the summary')
print(regression_model.summary())

# Evaluating
print("""\n\n\n
-----------------
Evaluating:
-----------------
""")
print('(1). Make predictions')
x_test_tilde = sm.add_constant(x_test)
y_test_pred = regression_model.predict(x_test_tilde)

x_new = np.array([65, 70, 75])  # Feature values of new instances
x_new_tilde = sm.add_constant(x_new)  # Preprocess x_new
y_pred = regression_model.predict(x_new_tilde)  # Predict the target

# Evaluate the model with Numpy
mse0 = np.mean((y_test-y_test_pred)**2)  # Mean Squared Error
rmse0 = np.sqrt(np.mean((y_test-y_test_pred)**2))  # Root Mean Squared Error
mae0 = np.mean(np.fabs(y_test-y_test_pred))  # Mean Absolute Error

# Evaluate the model using scikit-learn
mse = mean_squared_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

assert mse0 == mse
assert rmse0 == rmse
assert mae0 == mae

print(f"""\n\n\n
(2). Evaluate the model
    MSE      : {mse}
    RMSE     : {rmse}
    MAE      : {mae}
    R-SQUARED: {r2}
""")

print('\n\n\n(3). Present the model graphically')
plt.scatter(x_train, y_train, s=7)
plt.plot(x_train, beta_0 + beta_1 * x_train)
plt.scatter(x_test, y_test, c='maroon', s=25)
plt.scatter(x_test, y_test_pred, c='red', alpha=0.5, s=50)
plt.scatter(x_new, y_pred, c='black', s=100, alpha=0.25, lw=1)
plt.ylabel('Height')
plt.xlabel('Parent Height')
plt.tight_layout()

plt.legend(['Train set', 'Model', 'Test set', 'Test prediction',
            'New prediction'])
plt.show()
