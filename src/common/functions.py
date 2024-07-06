from common.np import *




def softmax(x):
    x = x - np.max(x, axis = 1, keepdims = True)
    out = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return out

def cov_one_hot(label:np.ndarray):
    data_size = label.shape[0]
    one_hot = np.zeros(shape = (label.shape[0],label.max() + 1))
    one_hot[np.arange(label.shape[0]),label] = 1
    return one_hot





def cross_entropy(t,y): #t is one-hot vector
    if y.ndim == 1 and t.ndim == 1: # when y and t is 1 dimension
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if y.size == t.size:
        t = t.argmax(axis = 1) #column index of 1

    batch_size = y.shape[0]

    entropy = -np.sum(np.log(y[np.arange(batch_size),t] + 1e-9))/batch_size
    return entropy