"""
This model calculate the height of a person given the height of the parents.
"""
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score)
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# Reading the data
print('1. Get an overview of the data')
df = pd.read_csv('data/heights_two_feature.csv')
print(df.head(10))

print('\n\n\nCheck missing values: ')
print(df.info())

# Visualizing data
print('\n\n\nVisualizing the data ')
plt.figure()
plt.scatter(df['Mother'], df['Height'], alpha=0.5, c='Red')
plt.scatter(df['Father'], df['Height'], alpha=0.5, c='Blue')
plt.ylabel('Height')
plt.xlabel('Parent Height')
plt.tight_layout()
plt.legend(['Mother Influence', 'Father influence'])

# x and y
print('\n\n\n2. Separate the data into x and y')
x, y = df[['Father', 'Mother']], df['Height']

# training and testing sets
print('\n\n\nSeparate the data into training and testing set')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Modeling
print('\n\n\n3. Modeling')
x_train_tilde = sm.add_constant(x_train)
print(x_train_tilde.head())
regression_model = sm.OLS(y_train, x_train_tilde).fit()

beta = regression_model.params
for i, b in enumerate(beta):
    print(f'beta_{i} is: {b}')
print(regression_model.summary())

# Making predictions
print('4. Make predictions')
x_test_tilde = sm.add_constant(x_test)
y_test_pred = regression_model.predict(x_test_tilde)

x_new = np.array([[65, 62], [70, 65], [75, 70]])  # New instances
x_new_tilde = sm.add_constant(x_new)  # Preprocess x_new
y_pred = regression_model.predict(x_new_tilde)  # Predict the target
print('Predictions:', y_pred)

# Evaluating
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

plt.show()
