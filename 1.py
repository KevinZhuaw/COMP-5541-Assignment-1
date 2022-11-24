from sklearn.datasets import load_iris
import numpy as np
from sklearn.naive_bayes import GaussianNB


class NBC:


    def __init__(self,X,y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self,X,y):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for classes in range(self.num_classes):
            X_c = X[y==classes]

            self.classes_mean[str(classes)] = np.mean(X_c, axis=0)
            self.classes_variance[str(classes)] = np.var(X_c, axis=0)
            self.classes_prior[str(classes)] = X_c.shape[0]/X.shape[0]

    def density_function(self, X, mean, sigma):
        constant = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps))
        probability = 0.5 * np.sum(np.power(X-mean, 2) / (sigma + self.eps), 1)
        return constant - probability

    def predict(self,X):

        probability = np.zeros((self.num_examples, self.num_classes))

        for classes in range(self.num_classes):

            prior = self.classes_prior[str(classes)]
            probability_c = self.density_function(X, self.classes_mean[str(classes)], self.classes_variance[str(classes)])
            probability[:,classes] = probability_c +np.log(prior)
            return np.argmax(probability, 1)




iris = load_iris()
X, y = iris['data'], iris['target']


N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]




nbc = NBC(Xtrain,ytrain)
nbc.fit(Xtrain,ytrain)
nbc1 = NBC(Xtest,ytest)
nbc1.fit(Xtest,ytest)
yhat = nbc1.predict(Xtest)
test_accuracy = np.mean(yhat == ytest)
print()





GNB = GaussianNB()
GNB.fit(Xtrain,ytrain)
y_pred = GNB.predict(Xtest)

# Course: COMP 5541
# Name: ZHUANG Kaiwen
# Student Number: 21066293g
