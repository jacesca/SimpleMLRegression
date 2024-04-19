"""This model predict the price of a house given its area in square feets."""
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score)
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# Reading the data
print('1. Get an overview of the data')
df = pd.read_csv('data/houses_simple.csv')
print(df.head(10))

print('\n\n\nCheck missing values: ')
print(df.info())

print('\n\n\nVisualizing the data ')
plt.figure()
plt.scatter(df['square_feet'], df['price'], alpha=0.5)
plt.ylabel('Price ($)')
plt.xlabel('Square Feet')
plt.tight_layout()


print('\n\n\n2. Separate the data into x and y')
x, y = df['square_feet'], df['price']

print('\n\n\nSeparate the data into training and testing set')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print('\n\n\n3. Modeling')
x_train_tilde = sm.add_constant(x_train)
regression_model = sm.OLS(y_train, x_train_tilde).fit()

beta_0, beta_1 = regression_model.params
print('beta_0 is', beta_0)
print('beta_1 is', beta_1)
print(regression_model.summary())

print('4. Make predictions')
x_test_tilde = sm.add_constant(x_test)
y_test_pred = regression_model.predict(x_test_tilde)

x_new = np.array([1300, 10000, 25000])  # Feature values of new instances
x_new_tilde = sm.add_constant(x_new)  # Preprocess x_new
y_pred = regression_model.predict(x_new_tilde)  # Predict the target

print('5. Evaluate the model')
# With Numpy
mse0 = np.mean((y_test-y_test_pred)**2)  # Mean Squared Error
rmse0 = np.sqrt(np.mean((y_test-y_test_pred)**2))  # Root Mean Squared Error
mae0 = np.mean(np.fabs(y_test-y_test_pred))  # Mean Absolute Error

# Using scikit-learn
mse = mean_squared_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

assert mse0 == mse
assert rmse0 == rmse
assert mae0 == mae

print(f"""
    MSE      : {mse}
    RMSE     : {rmse}
    MAE      : {mae}
    R-SQUARED: {r2}
""")

print('\n\n\n6. Present the model graphically')
plt.figure()
plt.scatter(x_train, y_train, s=7)
plt.plot(x_train, beta_0 + beta_1 * x_train)
plt.scatter(x_test, y_test, c='maroon', s=25)
plt.scatter(x_test, y_test_pred, c='red', alpha=0.5, s=50)
plt.scatter(x_new, y_pred, c='black', s=100)
plt.ylabel('Price ($)')
plt.xlabel('Square Feet')
plt.tight_layout()

plt.legend(['Train set', 'Model', 'Test set', 'Test prediction',
            'New prediction'])
plt.show()
