from cifar10_web import cifar10
import numpy as np
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = cifar10(path=None)

image = train_images[1, 0:3072] # or image[0]

print(train_labels[1])
print(train_labels[47188])

image = image.reshape(3, 32, 32)
image = image.transpose(1, 2, 0)

red_image = image[0, 0:32, 0:32] # or image[0]

plt.imshow(image)
plt.show()

print("happy world!")

image01 = train_images[30000, 0:3072]
print(train_labels[30000])

image01 = image01.reshape(3, 32, 32)
image01 = image01.transpose(1, 2, 0)

plt.imshow(image01)
plt.show()

print("happy world!")