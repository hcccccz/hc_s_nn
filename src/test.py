from common.layers import Affine
from data.mnist import load_mnist
import numpy as np


(x_train, t_train), (x_test, t_test) = load_mnist(dataset_dir="/home/hc/Documents",normalize=True, one_hot_label=True)



affine_W = np.random.randn(784,100)
affine_b = np.zeros(100)
affine = Affine(affine_W, affine_b)
x1 = affine.forward(x_train)

print(x1.shape)
print(x1.mean(axis=0).shape)

#training
mu = x1.mean(axis = 0)
xc = x1 - mu
var = np.mean(xc ** 2, axis = 0)
std = np.sqrt(var + 10e-7)
xn = xc / std
