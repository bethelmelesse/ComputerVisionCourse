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
        # pre_all = []
        all_result = np.zeros((X.shape[0],10), dtype = int)
        print(all_result)
        for i in range(X.shape[0]):
            print(i)
            # print(X.shape)
            d1 = X[i] - self.Xtr
            # print(d1.shape)
            # print(" ")
            d1 = np.absolute(d1)
            d1 = np.sum(d1, axis=1)
            # print(d1)
            result = np.argmin(d1)
            # print(result)
            pre = self.Ytr[result]
            # print(pre)
            # pre_all.append(pre)
            all_result[i] = pre
            print(pre)

            image = X[i].reshape(3, 32, 32)
            image = image.transpose(1, 2, 0)
            plt.imshow(image)

            closest_train_image = Xtr[result]
            image = closest_train_image.reshape(3, 32, 32)
            image = image.transpose(1, 2, 0)
            plt.figure()
            plt.imshow(image)
            plt.show()
        return all_result


def calc_accuracy(predict, gt):
    predict = np.argmax(predict, axis=1)
    gt = np.argmax(gt, axis=1)
    return np.mean(predict == gt) * 100.0


nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
nn.train(Xtr, Ytr)  # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte[0:10])  # predict labels on the test images

# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
accuracy = calc_accuracy(Yte_predict, Yte[0:10])
print('accuracy: ', accuracy)
print(Yte_predict)