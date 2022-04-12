from cifar10_web import cifar10
import numpy as np

train_images, train_labels, test_images, test_labels = cifar10(path=None)

def predict(W, b, X):
    # TODO: Fill out this function, make it return the correct Y = XW+b
    # X is of shape N*D1 array, W is D1*D2 array, b is D2 array.

    # return Y
    return None


def l1_loss(W, b, X, Y):
    # TODO: Return l1 loss for Y = XW+b
    # X is of shape N*D1 array, W is D1*D2 array, b is D2 array, Y is N*D2 array.

    # return Y
    return None

def svm_loss(W, b, X, Y):
    # TODO: Return l1 loss for Y = XW+b
    # X is of shape N*D1 array, W is D1*D2 array, b is D2 array, Y is N*D2 array.

    # return Y
    return None

np.random.seed(12345)
W = np.random.rand(train_images.shape[0], test_images.shape[1]).astype(train_images.dtype)
b = np.random.rand(test_images.shape[1]).astype(train_images.dtype)
ret_y = predict(W, b, train_images)
print('linear predictions -', ret_y)

# ret_y = l1_loss(W, b, train_images, test_images)
# print('linear predictions -', ret_y)
# ret_y = svm_loss(W, b, train_images, test_images)
# print('linear predictions -', ret_y)