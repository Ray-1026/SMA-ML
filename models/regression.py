import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


class Regression:
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y
        self.weights = None
        self.test_loss = None
        self.train_loss = None

    def fit(self):
        """
        Formula : betas = (X.T * X)^-1 * X.T * y
        K-fold cross validation
        """
        kf = KFold(n_splits=5, shuffle=True)

        for fold, (train_x, valid_x) in enumerate(kf.split(self.X)):
            regression = LinearRegression().fit(self.X[train_x], self.y[train_x])
            weights = np.array([regression.intercept_, *regression.coef_])
            # X_prime = np.hstack([np.ones((train_x.shape[0], 1)), self.X[train_x]])
            # weights = (
            #     np.linalg.inv(X_prime.T.dot(X_prime))
            #     .dot(X_prime.T)
            #     .dot(self.y[train_x])
            # )

            if fold == 0:
                self.weights = weights
                self.train_loss = self.rmse_loss(
                    self.X[train_x], self.y[train_x], weights
                )
                self.test_loss = self.rmse_loss(
                    self.X[valid_x], self.y[valid_x], weights
                )
            else:
                self.weights += weights
                self.train_loss += self.rmse_loss(
                    self.X[train_x], self.y[train_x], weights
                )
                self.test_loss += self.rmse_loss(
                    self.X[valid_x], self.y[valid_x], weights
                )

            # print(f"Fold {fold + 1} : {self.loss}")

        self.weights /= 5
        self.train_loss /= 5
        self.test_loss /= 5

        self.ensemble_model = LinearRegression()
        self.ensemble_model.coef_ = self.weights[1:]
        self.ensemble_model.intercept_ = self.weights[0]

        # joblib.dump(self.regression, f"weights/{self.name}.pkl")

    def predict(self, X, beta):
        """
        Formula : y = X * betas
        """
        X_prime = np.hstack([np.ones((X.shape[0], 1)), X])
        # print((X_prime @ beta).shape)
        return X_prime @ beta

    def rmse_loss(self, X, y, beta):
        """
        root mean square error
        """
        return np.sqrt(np.mean((y - self.predict(X, beta)) ** 2))
