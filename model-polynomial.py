"""
This is a sample model of polynomial regression.

Predicting values outside the training set's range is called **extrapolation**,
and predicting values inside the range is **interpolation**.
The Regression does not handle the extrapolation well. It is used for
interpolation and can yield absurd predictions when new instances are out
of the training set's range.

**Overfitting** is when the built model is too complex so that it can perfectly
fit the training data, but it does not predict unseen instances that well.
Good metrics on the training set
Bad metrics on the test set

**Underfitting** is when the built model is too simple that it does not even
fit the training data well. In that cases, predictions of the unseen instances
are wrong too.
Bad metrics on the training set
Bad metrics on the test set

**Good fit**
Good metrics on the training set
Good metrics on the test set

"""
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
df = pd.read_csv('data/poly.csv')
print(df.head())
print(df.info())

# Visualizing the data
plt.figure()
plt.scatter(df['Feature'], df['Target'], alpha=0.5)
plt.ylabel('Target')
plt.xlabel('Feature')
plt.title('Data Overview')
plt.tight_layout()

# Separating the data into x and y
x = df[['Feature']]  # or df.Feature.reshape(-1, 1)
y = df['Target']

# Separating the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=0)

# Preparing the visualization for each polynomial
colors = ['purple', 'magenta', 'darkorange', 'darkgoldenrod', 'forestgreen']
plt.figure()
plt.subplot(3, 2, 1)
plt.scatter(df['Feature'], df['Target'], alpha=0.5, c='blue')
plt.ylabel('Target')
plt.xlabel('Feature')
plt.title('Data Overview')

degrees = [2, 3, 4, 5, 6]  # Polynomial degree: 2 >> y = a + bx + cx^2
alpha = 0.05
x_new = np.linspace(-0.1, 1.5, 80)
for i, n in enumerate(degrees):
    # Creating a high degree polynoial regression y = a + bx + cx^2
    # df['Feature_squared'] = df.Feature ** 2  # and then >> x_tilde = sm.add_constant(x)  # noqa
    # or
    poly = PolynomialFeatures(n)
    x_train_tilde = poly.fit_transform(x_train)

    # Initialize the OLS object and train it
    regression_model = sm.OLS(y_train, x_train_tilde).fit()
    # print(f'\n\nModel Degree {n}\n: {regression_model.params}')
    # print(regression_model.summary())

    # Evaluating the model
    x_test_tilde = poly.fit_transform(x_test)
    y_test_pred = regression_model.predict(x_test_tilde)

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"""
    \nEvaluation of Model PolynomialRegression Deg. {n}
        MSE      : {mse}
        RMSE     : {rmse}
        MAE      : {mae}
        R-SQUARED: {r2}
    """)

    # Predicting new values
    x_new_tilde = poly.fit_transform(x_new.reshape(-1, 1))
    y_new_pred = regression_model.predict(x_new_tilde)

    # Getting the confidence interval
    lower = regression_model.get_prediction(x_new_tilde).summary_frame(alpha)['mean_ci_lower']  # noqa
    upper = regression_model.get_prediction(x_new_tilde).summary_frame(alpha)['mean_ci_upper']  # noqa

    # Visualizing the model
    plt.subplot(3, 2, i + 2)
    plt.scatter(df['Feature'], df['Target'], alpha=0.5, c='blue')
    plt.axvspan(x.values.min(), x.values.max(), alpha=0.5, color='bisque')
    plt.plot(x_new, y_new_pred, c=colors[i], alpha=0.5)
    plt.fill_between(x_new, lower, upper, color=colors[i], alpha=0.4)
    plt.ylabel('Target')
    plt.xlabel('Feature')
    plt.title(f'Model Degree {n}')

plt.tight_layout()
plt.show()
