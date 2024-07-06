import sys, os
sys.path.append(os.pardir)
from data.mnist import load_mnist
from hc_s_nn import Network
from common.trainer import Trainer
from common.optimizers import SGD
from sklearn.model_selection import train_test_split
from common.np import *
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
X_train, X_test, T_train, T_test = train_test_split(x_train, t_train)

X_train=np.asarray(X_train)
T_train=np.asarray(T_train)
X_test=np.asarray(X_test)
T_test=np.asarray(T_test)

# print(X_train.shape)
# network = TwoLayerNet(784,100,10)
network = Network(feature_size=784,hidden_size=100,layer_size=50,out_size=10)
optimizer = SGD()

trainer = Trainer(network=network, x_train=X_train, t_train=T_train,
                   x_test=X_test, t_test=T_test,
                  epochs = 100, mini_batch_size = 1000, optimizer = optimizer)
trainer.train()