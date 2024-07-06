from time import time
from common.np import *

class Trainer:

    def __init__(self, network, x_train, t_train, x_test, t_test,
                  epochs, mini_batch_size, optimizer):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.optimizer = optimizer
        """
        遍历多少次整个数据
        """
        self.batch_size = mini_batch_size


        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        """
        how many iteration required to cover the size of dataset
        """
        self.max_iter = int(epochs * self.iter_per_epoch)
        """
        完成一个epoch需要多次iter，完成整个epochs为max_iter
        """

        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []

    def train_step(self):
        t1 = time()
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)

        # "=---------------------------------------------"
        # print("grad:")
        # for grad in grads.keys():
        #     print(grad+":",end="")
        #     print(grads[grad].shape)

        # "----------------------------------------------"
        # print("prarm:")
        # for param in self.network.params:
        #     print(param+":",end="")
        #     print(self.network.params[param].shape)
        # "----------------------------------------------"


        self.optimizer.update(self.network.params,grads = grads)
        loss = self.network.loss(x_batch, t_batch)



        self.train_loss_list.append(loss)

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

        train_accuracy = self.network.accuracy(self.x_train, self.t_train)

        test_accuracy = self.network.accuracy(self.x_test, self.t_test)

        self.current_iter += 1
        t2 = time()
        print("iter: {}, epoch: {}, loss: {}, train acc: {}, test acc: {}, time_iter: {}".format(self.current_iter,
                                                                                  self.current_epoch, loss, train_accuracy, test_accuracy, t2 - t1))
    def train(self):
        for i in range(self.max_iter):
            self.train_step()