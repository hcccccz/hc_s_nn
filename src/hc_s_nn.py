from common.np import *
from collections import OrderedDict
from common.layers import Affine, SoftmaxWithLoss, Relu



class Network:

    def __init__(self, feature_size, hidden_size_list, out_size): #lauyer_size >= 2

        self.activation_out ={}
        self.params = {}
        self.layers = OrderedDict()

        self.feature_size = feature_size
        self.out_size = out_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(self.hidden_size_list)

        self.__init__weight("Relu")

        for idx in range(1, self.hidden_layer_num + 1):
            """
            hidden_layer_num = 3
            1, 2, 3
            """
            self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)],
                                                      self.params["b" + str(idx)])
            self.layers["activate" + str(idx)] = Relu()

        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(self.params["W" + str(idx)],
                                                  self.params["b" + str(idx)])



        self.lastlayer = SoftmaxWithLoss()

    def __init__weight(self, weight_init_std):
        """
        默认3层hidden 每层100
        all_size_list 就有 [784, 100, 100, 100, 10] = 5
        (784, 100)
        (100,100)
        (100,100)
        (100,10)

        """

        all_size_list = [self.feature_size] + self.hidden_size_list + [self.out_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std

            if str(scale).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(scale).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])


    def predict(self, x):
        for layer_name in self.layers.keys():
            x = self.layers[layer_name].forward(x)
            if layer_name.startswith("Relu"):
                self.activation_out[layer_name] = x


        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastlayer.forward(t, y)

        return loss

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.lastlayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        for layer_name in self.layers.keys():
            if layer_name.startswith("Affine"):

                grads["W"+layer_name.replace("Affine","")], grads["b"+layer_name.replace("Affine","")] = self.layers[layer_name].dW, self.layers[layer_name].db


        return grads

    def get_act_out(self):
        return self.activation_out

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

if __name__ == '__main__':

    net = Network(784,10,4,10)
    print(net.params.keys())
    print(net.test().keys())