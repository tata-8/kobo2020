import numpy as np
from common.functions import softmax

def cross_entropy_error(y, t):
   if y.ndim == 1:
       t = t.reshape(1, t.size)
       y = y.reshape(1, y.size)
   # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
   if t.size == y.size:
       t = t.argmax(axis=1)
   batch_size = y.shape[0]
   return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


class SoftmaxWithLoss:
   def __init__(self):
       self.loss = None
       self.y = None # softmaxの出力
       self.t = None # 教師データ
   def forward(self, x, t):
       self.t = t
       self.y = softmax(x)
       # forwardの式
       # -sum ( t * log (y))
       self.loss = cross_entropy_error(self.y, self.t)
       return self.loss
   def backward(self, dout=1):
       # backwardの式
       # yi - ti (iはIndex)
       batch_size = self.t.shape[0]
       # Backwardを実装して、微分値をdxに代入してください
       dx = (self.y - self.t) / batch_size
       return dx

from collections import OrderedDict
def numerical_grad(f, x):
   h = 1e-4 # 0.0001
   grad = np.zeros_like(x)
   it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
   while not it.finished:
       idx = it.multi_index
       tmp_val = x[idx]
       x[idx] = float(tmp_val) + h
       fxh1 = f(x) # f(x+h)
       x[idx] = tmp_val - h 
       fxh2 = f(x) # f(x-h)
       grad[idx] = (fxh1 - fxh2) / (2*h)
       x[idx] = tmp_val # 値を元に戻す
       it.iternext()   
   return grad
class TwoLayerNet:
   def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
       # 重みの初期化
       self.params = {}
       self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
       self.params['b1'] = np.zeros(hidden_size)
       self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
       self.params['b2'] = np.zeros(output_size)
       # レイヤの生成
       self.layers = OrderedDict()
       self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
       self.layers['Relu1'] = Relu()
       self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
       self.lastLayer = SoftmaxWithLoss()        
   def predict(self, x):
       for layer in self.layers.values():
           x = layer.forward(x)
       return x
   # x:入力データ, t:教師データ
   def loss(self, x, t):
       y = self.predict(x)
       return self.lastLayer.forward(y, t)
   def numerical_gradient(self, x, t):
       loss_W = lambda W: self.loss(x, t)
       grads = {}
       grads['W1'] = numerical_grad(loss_W, self.params['W1'])
       grads['b1'] = numerical_grad(loss_W, self.params['b1'])
       grads['W2'] = numerical_grad(loss_W, self.params['W2'])
       grads['b2'] = numerical_grad(loss_W, self.params['b2'])
       return grads

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

class Affine:
   def __init__(self, W, b):
       self.W =W
       self.b = b
       self.x = None
       self.original_x_shape = None
       # 重み・バイアスパラメータの微分
       self.dW = None
       self.db = None
   def forward(self, x):
       # テンソル対応
       self.original_x_shape = x.shape
       x = x.reshape(x.shape[0], -1)
       self.x = x
       out = np.dot(self.x, self.W) + self.b
       return out
   def backward(self, dout):
       dx = np.dot(dout, self.W.T)
       self.dW = np.dot(self.x.T, dout)
       self.db = np.sum(dout, axis=0)
       dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
       return dx


def gradient(network, x, t):
   # 自分で実装したSoftmax with lossクラスを使ってみてください
   lastLayer = SoftmaxWithLoss()
   # forward
   #self.loss(x, t)
   y = network.predict(x)
   lastLayer.forward(y, t)
   # backward
   dout = 1
   dout = lastLayer.backward(dout)
   #layers = list(self.layers.values())
   layers = list(network.layers.values())
   layers.reverse()
   for layer in layers:
      dout = layer.backward(dout)
   # 設定
   grads = {}
   #grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
   grads['W1'], grads['b1'] = network.layers['Affine1'].dW, network.layers['Affine1'].db
   #grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
   grads['W2'], grads['b2'] = network.layers['Affine2'].dW, network.layers['Affine2'].db
   return grads


if __name__ == '__main__':
    from dataset.mnist import load_mnist
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:3]
    t_batch = t_train[:3]
    # 数値微分
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    # Backward
    #grad_backprop = gradient(x_batch, t_batch)
    grad_backprop = gradient(network, x_batch, t_batch)
    for key in grad_numerical.keys():
        diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
        print(key + ":" + str(diff))



#実行結果
# % python my_softmaxWithLoss.py
# W1:2.4044474026310363e-13
# b1:1.079465840987063e-12
# W2:9.400041980152547e-13
# b2:1.2034818003270332e-10

#感想
# とりあえず実行できたと思います。
#　バックプロパゲーションと数値微分の差も小さいので成功だと思います。