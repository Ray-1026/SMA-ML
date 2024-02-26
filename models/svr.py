import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR


class SVREnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)


class SVR_Model:
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y
        self.weights = None
        self.test_loss = None
        self.train_loss = None

        self.models = []

    def fit(self):
        kf = KFold(n_splits=5, shuffle=True)

        self.test_loss = 0
        self.train_loss = 0

        for _, (train_x, valid_x) in enumerate(kf.split(self.X)):
            svr = SVR(kernel="rbf")
            svr.fit(self.X[train_x], self.y[train_x].ravel())

            self.models.append(svr)
            self.train_loss += self.rmse_loss(
                self.X[train_x], self.y[train_x], [self.models[-1]]
            )
            self.test_loss += self.rmse_loss(
                self.X[valid_x], self.y[valid_x], [self.models[-1]]
            )

        self.train_loss /= 5
        self.test_loss /= 5

        self.ensemble_model = SVREnsemble(self.models)

    def predict(self, X, models):
        y_pred = None
        for i in models:
            if y_pred is None:
                y_pred = i.predict(X)
            else:
                y_pred += i.predict(X)

        y_pred /= len(models)
        return y_pred

    def rmse_loss(self, X, y, models):
        """
        root mean square error
        """
        return np.sqrt(np.mean((y - self.predict(X, models)) ** 2))
