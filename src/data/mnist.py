import numpy as np
import os
import pickle

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _load_label(dataset_dir, file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels



def _load_img(dataset_dir, file_name):
    file_path = dataset_dir + "/" + file_name
    print("Converting " + file_name + " to NumPy Array ...")
    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, img_size)
    print("Done")
    return data





def _convert_numpy(dataset_dir, key_file):
    dataset = {}
    dataset['train_img'] =  _load_img(dataset_dir, key_file['train_img'])
    dataset['train_label'] = _load_label(dataset_dir, key_file['train_label'])
    dataset['test_img'] = _load_img(dataset_dir, key_file['test_img'])
    dataset['test_label'] = _load_label(dataset_dir, key_file['test_label'])

    return dataset



def init_mnist(dataset_dir=os.path.dirname(os.path.abspath(__file__))):
    key_file = {
    'train_img':'train-images.idx3-ubyte',
    'train_label':'train-labels.idx1-ubyte',
    'test_img':'t10k-images.idx3-ubyte',
    'test_label':'t10k-labels.idx1-ubyte'
    }


    dataset = _convert_numpy(dataset_dir, key_file)
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T
def load_mnist(dataset_dir = os.path.dirname(os.path.abspath(__file__)), normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
        init_mnist(dataset_dir)

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

if __name__ == "__main__":
    init_mnist()