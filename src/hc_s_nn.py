from common.np import *
from collections import OrderedDict
from common.layers import Affine, SoftmaxWithLoss, Relu



class Network:

    def __init__(self, feature_size, hidden_size, layer_size, out_size): #lauyer_size >= 2

        self.params = {}
        self.layers = OrderedDict()

        self.params["w0"] = 0.01 * np.random.randn(feature_size, hidden_size)
        self.params["b0"] = np.zeros(hidden_size)

        self.layers["Affine0"] = Affine(self.params["w0"], self.params["b0"])
        self.layers["Relu0"] = Relu()
        for i in range(1,layer_size - 1):
            self.params["w"+str(i)] = 0.01 * np.random.randn(hidden_size, hidden_size)
            self.params["b"+str(i)] = np.zeros(hidden_size)

            self.layers["Affine"+str(i)] = Affine(self.params["w"+str(i)], self.params["b"+str(i)])
            self.layers["Relu"+str(i)] = Relu()

        self.params["w"+str(layer_size-1)] = 0.01 * np.random.randn(hidden_size, out_size)
        self.params["b"+str(layer_size-1)] = np.zeros(out_size)
        self.layers["Affine"+str(layer_size-1)] = Affine(self.params["w"+str(layer_size-1)], self.params["b"+str(layer_size-1)])

        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():

            x = layer.forward(x)

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

                grads["w"+layer_name.replace("Affine","")], grads["b"+layer_name.replace("Affine","")] = self.layers[layer_name].dW, self.layers[layer_name].db


        return grads

    def test(self):
        for layer_name in self.layers.keys():
            if layer_name.startswith("Affine"):
                print(self.layers[layer_name].W.shape)
    # def print_struct(self):


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