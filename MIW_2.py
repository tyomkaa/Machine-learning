from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # konfiguruje generator znaczników i mapę kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # rysuje wykres powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # rysuje wykres wszystkich próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl, edgecolor='black')


# class Perceptron(object):

#     # Konstruktor, podajemy współczynik uczenia sie oraz ilość epok
#     def __init__(self, eta=0.01, n_iter=10):
#         self.eta = eta
#         self.n_iter = n_iter

#     def fit(self, X, y):
#         self.w_ = np.zeros(1+ X.shape[1])
#         self.errors_ = []

#         for _ in range(self.n_iter):
#             errors = 0
#             for xi, target in zip(X,y):
#                 update = self.eta * (target - self.predict(xi))
#                 self.w_[1:] += update *xi
#                 self.w_[0] += update
#                 errors += int(update != 0.0)
#             self.errors_.append(errors)
#         return self

#     def net_input(self, X):
#         return np.dot(X, self.w_[1:]) + self.w_[0]

#     def predict(self, X):
#         return np.where(self.net_input(X) >= 0.0, 1, -1)
    
# class MulticlassPerceptron(object):
#     def __init__(self, eta=0.001, n_iter=10000):
#         self.eta = eta
#         self.n_iter = n_iter
    
#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
#         self.classifiers_ = []
#         for c in self.classes_:
#             y_binary = np.where(y == c, 1, -1)
#             classifier = Perceptron(self.eta, self.n_iter)
#             classifier.fit(X, y_binary)
#             self.classifiers_.append(classifier)    
    
#     def predict(self, X):
#         scores = np.zeros((X.shape[0], len(self.classes_)))
#         for i, classifier in enumerate(self.classifiers_):
#             scores[:, i] = classifier.net_input(X)
#         return self.classes_[np.argmax(scores, axis=1)]

    

# def main():
#     # pobiera danne do uczenia i testowania
#     iris = datasets.load_iris()
#     X = iris.data[:, [2, 3]]
#     y = iris.target
#     # podział danych na testowe i treningowe
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#     clf = MulticlassPerceptron(eta=0.1, n_iter=10)
#     clf.fit(X_train, y_train)

#     # wyświetla wykres
#     plot_decision_regions(X=X_train, y=y_train, classifier=clf)
#     plt.xlabel(r'$x_1$')
#     plt.ylabel(r'$x_2$')
#     plt.legend(loc='upper left')
#     plt.show()

#     y_pred = clf.predict(X_test)
#     print('\nAccuracy:', accuracy_score(y_test, y_pred))

    

# if __name__ == '__main__':
#     main()


import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class MultiClassLogisticRegression(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.classifiers_ = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            binary_y = np.where(y == c, 1, 0)
            lr = LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state)
            lr.fit(X, binary_y)
            self.classifiers_[c] = lr
        return self

    def predict(self, X):
        output = np.zeros((X.shape[0], len(self.classifiers_)))
        for i, c in enumerate(self.classifiers_):
            output[:, i] = self.classifiers_[c].net_input(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        output = np.zeros((X.shape[0], len(self.classifiers_)))
        for i, c in enumerate(self.classifiers_):
            output[:, i] = self.classifiers_[c].activation(self.classifiers_[c].net_input(X))
        return output


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    mclr = MultiClassLogisticRegression(eta=0.05, n_iter=10000, random_state=1)
    mclr.fit(X_train, y_train)
    plot_decision_regions(X=X_train, y=y_train, classifier=mclr)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()
    print('\nPredict proba is:\n', mclr.predict_proba(X_test)) 


if __name__ == '__main__':
    main()