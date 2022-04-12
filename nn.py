from cifar10_web import cifar10
import numpy as np
from matplotlib import pyplot as plt

Xtr, Ytr, Xte, Yte = cifar10(path=None)


class NearestNeighbor(object):
    def __init__(self):
        self.Ytr = None
        self.Xtr = None

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.Ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        print(X.shape)  # (10000, 3072)
        print(self.Xtr.shape)  # (50000, 3072)
        print(" ")

        for i in range(X.shape[0]):
            d1 = X[i] - self.Xtr  # test image will be subtracted from all the training image - one at a time (broadcast)
            # print(d1)
            d1 = np.absolute(d1)
            # print(d1)
            d1 = np.sum(d1, axis=1)
            # print(d1)
            min_index = np.argmin(d1)
            # print(min_index)
            predict = Ytr[min_index]
            print(predict)

            test_image = X[i].reshape(3, 32, 32)
            test_image = test_image.transpose(1, 2, 0)
            plt.imshow(test_image)
            plt.show()


        return None


nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
nn.train(Xtr, Ytr)  # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte[0:5])  # predict labels on the test images
print(Yte_predict)
