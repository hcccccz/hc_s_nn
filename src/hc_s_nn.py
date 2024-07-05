from common.np import *
from collections import OrderedDict
from common.layers import Affine, SoftmaxWithLoss, Relu



class Network:

    def __init__(self, feature_size, hidden_size, layer_size, out_size):

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
        self.params["b"+str(layer_size-1)] = np.zeros(hidden_size)
        self.layers["Affine"+str(layer_size-1)] = Affine(self.params["w1"], self.params["b1"])

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


        grads["w1"], grads["b1"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["w2"], grads["b2"] = self.layers["Affine2"].dW, self.layers["Affine2"].db
        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

if __name__ == '__main__':

    net = Network(784,10,4,10)

    print(net.layers.keys())