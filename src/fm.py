from initializer import TParametrInitializers
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.sparse import csc_matrix
import os


def err(x):
    return np.power(x, 2).mean() ** 0.5


def train_test_split(X, y, test_fraction=0.2):
    X = X.tocsr()
    permutation = np.random.permutation(X.shape[0])
    validation_idx = permutation[:round(X.shape[0] * test_fraction)]
    train_idx = permutation[round(X.shape[0] * test_fraction):]
    X_test, y_test = X[validation_idx], y[validation_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    X_train = X_train.tocsc()
    return X_train, X_test, y_train, y_test


class FM:

    def __init__(self, n_iter=1, latent_dimension=5,
                 lambda_w=0.065, lambda_v=0.065):
        self.latent_dimension = latent_dimension  # Latent dimension
        self.test_fraction = 0.05

        self.lambda_w0 = 0
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.n_iter = n_iter
        if os.path.exists('./data/validate_result.txt'):
            os.remove('./data/validate_result.txt')

        self.w0, self.w, self.V = TParametrInitializers(self.latent_dimension).xavier()

    def fit(self, X: csc_matrix, y):
        print("Fit starts")

        # X, X_, y, y_ = train_test_split(X, y, self.test_fraction)
        e = self.predict(X) - y
        q = X.dot(self.V)
        n_samples, n_features = X.shape

        X = X.tocsc()
        for i in range(self.n_iter):
            # self.evaluate(i, X_, y_)

            # Global bias
            w0_ = - (e - self.w0).sum() / (n_samples + self.lambda_w0)
            e += w0_ - self.w0
            self.w0 = w0_
            # self.evaluate(i, X_, y_)

            # 1-way interaction
            for l in range(n_features):
                Xl = X.getcol(l).toarray()
                print(("\r Iteration #{} 1-way interaction "
                       "progress {:.2%}; train error {}").format(i, l / n_features, err(e)), end="")
                w_ = - ((e - self.w[l] * Xl) * Xl).sum() / (np.power(Xl, 2).sum() + self.lambda_w)
                e += (w_ - self.w[l]) * Xl
                self.w[l] = w_
            # self.evaluate(i, X_, y_)

            # 2-way interaction
            for f in range(self.latent_dimension):
                Qf = q[:, f].reshape(-1, 1)
                for l in range(n_features):
                    Xl = X.getcol(l)
                    idx = Xl.nonzero()[0]
                    Xl = Xl.data.reshape(-1, 1)
                    Vlf = self.V[l, f]
                    print(("\r Iteration #{} 2-way interaction progress {:.2%};" +
                          "error {:.5}; validation_error NO").format(i, (f * n_features + l)
                                                                     / (self.latent_dimension * n_features),
                                                                     err(e)), end="")
                    h = Xl * Qf[idx] - np.power(Xl, 2) * Vlf
                    v_ = - ((e[idx] - Vlf * h) * h).sum() / (np.power(h, 2).sum() + self.lambda_v)
                    e[idx] += (v_ - Vlf) * h
                    Qf[idx] += (v_ - Vlf) * Xl
                    self.V[l, f] = v_
                q[:, f] = Qf.reshape(-1)
            # self.evaluate(i, X_, y_)

    def predict(self, X):
        X = X.tocsr()
        n_samples = X.shape[0]
        result = np.full((n_samples, 1), self.w0)
        result += X.dot(self.w.reshape(-1, 1))
        result += (np.power(X.dot(self.V), 2) - X.power(2).dot(np.power(self.V, 2))).sum(axis=1).reshape(-1, 1) / 2
        return result

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return mean_squared_error(y_pred, y_true)

    def evaluate(self, i, X, y):
        with open("data/validate_result.txt", "a") as out:
            out.write('{}  {}\n'.format(i, self.score(X, y)))

