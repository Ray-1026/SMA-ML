import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression as Regression


def read_tetra_csv(path):
    with open(path, newline="") as csvfile:
        reader = np.array(list(csv.reader(csvfile)))
        features = reader[1:, 17:25].astype(float)
        parameters = reader[1:, 25:27].astype(float)

    return features, parameters


def read_mono_csv(path):
    with open(path, newline="") as csvfile:
        reader = np.array(list(csv.reader(csvfile)))
        features = reader[1:, 17:25].astype(float)
        parameters = reader[1:, 25:29].astype(float)

    return features, parameters


def read_B2_test(path):
    with open(path, newline="") as csvfile:
        reader = np.array(list(csv.reader(csvfile)))
        features = reader[1:, 6:14].astype(float)
        parameters = reader[1:, 14].astype(float)

    return features, parameters


def read_B19_test(path):
    with open(path, newline="") as csvfile:
        reader = np.array(list(csv.reader(csvfile)))
        features = reader[1:, 6:14].astype(float)
        parameters = reader[1:, 14:18].astype(float)

    return features, parameters


def exhaustion(n, k, at, temp_list, lists):
    if len(temp_list) == k:
        lists.append(temp_list)
        return
    for i in range(at, n):
        if i not in temp_list:
            exhaustion(n, k, i, temp_list + [i], lists)


class LinearRegression:
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
            regression = Regression().fit(self.X[train_x], self.y[train_x])
            weights = np.array([regression.intercept_, *regression.coef_])
            # X_prime = np.hstack([np.ones((train_x.shape[0], 1)), self.X[train_x]])
            # weights = (
            #     np.linalg.inv(X_prime.T.dot(X_prime))
            #     .dot(X_prime.T)
            #     .dot(self.y[train_x])
            # )
            # print(weights)

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

    def predict(self, X, beta):
        """
        Formula : y = X * betas
        """
        X_prime = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_prime @ beta

    def rmse_loss(self, X, y, beta):
        """
        root mean square error
        """
        return np.sqrt(np.mean((y - self.predict(X, beta)) ** 2))


if __name__ == "__main__":
    train_xt, train_yts = read_tetra_csv(
        "nature_dataset/Training dataset_tetragonal.csv"
    )
    train_xm, train_yms = read_mono_csv(
        "nature_dataset/Training dataset_monoclinic.csv"
    )

    print("Dataset : tetragonal")
    print(f"\tx : {train_xt.shape}")
    print(f"\ty : {train_yts.shape}\n")
    print("Dataset : monoclinic")
    print(f"\tx : {train_xm.shape}")
    print(f"\ty : {train_yms.shape}\n")

    choose = []
    for i in range(1, 9):
        exhaustion(8, i, 0, [], choose)

    best_loss = float("inf")
    best_list = None
    best_LR = None

    for i in choose:
        LR_list = []

        LR_at = LinearRegression("a_t", train_xt[:, i], train_yts[:, 0])
        LR_ct = LinearRegression("c_t", train_xt[:, i], train_yts[:, 1])

        LR_at.fit()
        LR_ct.fit()

        LR_list.append(LR_at)
        LR_list.append(LR_ct)

        LR_am = LinearRegression("a_m", train_xm[:, i], train_yms[:, 0])
        LR_bm = LinearRegression("b_m", train_xm[:, i], train_yms[:, 1])
        LR_cm = LinearRegression("c_m", train_xm[:, i], train_yms[:, 2])
        LR_beta = LinearRegression("beta_m", train_xm[:, i], train_yms[:, 3])

        LR_am.fit()
        LR_cm.fit()
        LR_bm.fit()
        LR_beta.fit()

        LR_list.append(LR_am)
        LR_list.append(LR_bm)
        LR_list.append(LR_cm)
        LR_list.append(LR_beta)

        total_loss = 0
        for j in range(len(LR_list)):
            total_loss += LR_list[j].test_loss

        if total_loss < best_loss:
            best_loss = total_loss
            best_list = i
            best_LR = LR_list

    print(f"Best Parameters : {best_list}\n")
    print("Weights : ")
    for j in range(len(best_LR)):
        print(
            f"\t{best_LR[j].name:^6} : {best_LR[j].weights[0]:>8.4f} {best_LR[j].weights[1]:>8.4f} {best_LR[j].weights[2]:>8.4f} {best_LR[j].weights[3]:>8.4f} {best_LR[j].weights[4]:>8.4f} {best_LR[j].weights[5]:>8.4f} {best_LR[j].weights[6]:>11.4e}"
        )

    print("\nTest Loss : ")
    for j in range(len(best_LR)):
        print(f"\t{best_LR[j].name:^6} : {best_LR[j].test_loss:>7.4f}")

    print("\nTrain Loss : ")
    for j in range(len(best_LR)):
        print(f"\t{best_LR[j].name:^6} : {best_LR[j].train_loss:>7.4f}")
