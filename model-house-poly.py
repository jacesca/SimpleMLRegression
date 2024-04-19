"""This is a high degree polynomial model to predict the price of a house"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from matplotlib import pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Reading the data
print('Data')
df = pd.read_csv('data/houses_poly.csv')
print(df.head())
print(df.info())

# Visualizing the data
plt.figure()
plt.scatter(df['age'], df['price'], alpha=0.5)
plt.ylabel('price')
plt.xlabel('age')
plt.title('Data Overview')
plt.tight_layout()

# Separating the data into x and y
x = df[['age']]  # or df.age.reshape(-1, 1)
y = df['price']

# Separating the data into testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=0)

# Creating a high degree polynoial regression model
n = 2  # Polynomial degree: 2 >> y = a + bx + cx^2
alpha = 0.05  # Coinfidence Interval
poly = PolynomialFeatures(n)
x_train_tilde = poly.fit_transform(x_train)

# Initialize the OLS object and train it
regression_model = sm.OLS(y_train, x_train_tilde).fit()
print(f'\n\nModel Degree {n}\n: {regression_model.params}')
print(regression_model.summary())

# Evaluating the model
x_test_tilde = poly.fit_transform(x_test)
y_test_pred = regression_model.predict(x_test_tilde)

mse = mean_squared_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f"""
\n\n\nModel Evaluation
    MSE      : {mse}
    RMSE     : {rmse}
    MAE      : {mae}
    R-SQUARED: {r2}
""")

# Predicting new values
x_new = np.linspace(0, 125, 200)
x_new_tilde = poly.fit_transform(x_new.reshape(-1, 1))
y_new_pred = regression_model.predict(x_new_tilde)

# Getting the confidence interval
lower = regression_model.get_prediction(x_new_tilde).summary_frame(alpha)['mean_ci_lower']  # noqa
upper = regression_model.get_prediction(x_new_tilde).summary_frame(alpha)['mean_ci_upper']  # noqa

# Visualizing the model
plt.figure()
plt.scatter(x, y, alpha=0.5, c='blue')
plt.axvspan(x.values.min(), x.values.max(), alpha=0.5, color='bisque')
plt.plot(x_new, y_new_pred, c='red')
plt.fill_between(x_new, lower, upper, color='yellow', alpha=0.5)
plt.ylabel('Target')
plt.xlabel('Feature')
plt.title(f'Model Degree {n}')
plt.tight_layout()
plt.legend(['Houses data', 'Interpolation Area', 'Model', 'Confidence Interval'])  # noqa

plt.show()
