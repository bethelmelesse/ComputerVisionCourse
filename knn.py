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

    def predict(self, X, k):
        """ X is N x D where each row is an example we wish to predict label for """
        print(X.shape)  # (10000, 3072)
        print(self.Xtr.shape)  # (50000, 3072)
        print(" ")
        # print(k)
        all_result = np.zeros((X.shape[0],10), dtype = int)

        for i in range(X.shape[0]):
            d1 = X[i] - self.Xtr  # test image will be subtracted from all the training image - one at a time (broadcast)
            d1 = np.absolute(d1)
            d1 = np.sum(d1, axis=1)
            # print(d1)

            sort_d1 = np.argsort(d1)
            # print(sort_d1)

            kth = sort_d1[0:k]
            # print(kth)

            predict = Ytr[kth]
            # print(predict)
            # print(" ")

            max_index = np.argmax(predict, axis=1)
            # print(max_index)                      # index of 1 of each row in the ndarray

            freq = np.bincount(max_index)         # count the occurrence of these numbers
            # print(freq)

            index_freq = np.argmax(freq)          # find the index of the freq occurring number
            # print(index_freq)

            final = predict[np.where(max_index == index_freq)]
            final2 = final[0]
            # print(final[0])

            all_result[i] = final2

            # print(" ")
            test_image = X[i].reshape(3, 32, 32)
            test_image = test_image.transpose(1, 2, 0)
            plt.imshow(test_image)
            # plt.show()

        return all_result

def calc_accuracy(predict, gt):
    predict = np.argmax(predict, axis=1)
    gt = np.argmax(gt, axis=1)
    return np.mean(predict == gt) * 100.0

nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
nn.train(Xtr, Ytr)  # train the classifier on the training images and labels
k = 5
Yte_predict = nn.predict(Xte[0:20], k)  # predict labels on the test images
print(Yte_predict)
accuracy = calc_accuracy(Yte_predict, Yte[0:20])
print(" ")
print(Yte[0:20])
print('Accuracy is', accuracy)
