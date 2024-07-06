from common.np import *
from common.functions import softmax, cross_entropy


class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b


        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx



class SoftmaxWithLoss():
    def __init__(self):
        #t is one-hot vector
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,t:np.ndarray,y:np.ndarray):
        self.y = softmax(y)
        self.t = t
        self.loss = cross_entropy(self.t,self.y)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) / batch_size
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

