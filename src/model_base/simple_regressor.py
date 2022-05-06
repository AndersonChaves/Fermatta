import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegressionStrategy:
    def __init__(self, X, y):
        self.X = np.array([[x] for x in X])
        self.y = np.array([[_y] for _y in y])

    def train(self):
        self.regression_line = LinearRegression().fit(self.X, self.y)

    # Expected 2d numpy array
    def predict(self, x):
        x = np.array([[_x] for _x in x])
        return self.regression_line.predict(x)


# Example
# X = np.array([[1], [3], [5], [7]])
# y = np.array([[10], [30], [50], [70]])
# r = LinearRegressionStrategy(X, y)
# r.train()
# print(r.predict(np.array([[20]])))

# reg = LinearRegression().fit(X, y)
# reg.score(X, y)
# print(reg.predict(np.array([[5]])))
