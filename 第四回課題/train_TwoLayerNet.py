import numpy as np
from dataset.mnist import load_mnist
from common.functions import softmax
from two_layer_net import TwoLayerNet

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







# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)
for i in range(iters_num):
   batch_mask = np.random.choice(train_size, batch_size)
   x_batch = x_train[batch_mask]
   t_batch = t_train[batch_mask]
   # 勾配
   #grad = network.numerical_gradient(x_batch, t_batch)
   #grad = gradient(x_batch, t_batch)
   grad = gradient(network, x_batch, t_batch)
   # 更新
   for key in ('W1', 'b1', 'W2', 'b2'):
       network.params[key] -= learning_rate * grad[key]
   loss = network.loss(x_batch, t_batch)
   train_loss_list.append(loss)
   if i % iter_per_epoch == 0:
       train_acc = network.accuracy(x_train, t_train)
       test_acc = network.accuracy(x_test, t_test)
       train_acc_list.append(train_acc)
       test_acc_list.append(test_acc)
       print("# ", train_acc, test_acc)


#実行結果
#  % python train_TwoLayerNet.py
#  0.11238333333333334 0.1087
#  0.90295 0.9088
#  0.9198666666666667 0.9232
#  0.9354333333333333 0.9359
#  0.9455 0.9434
#  0.9527333333333333 0.9489
#  0.9563833333333334 0.953
#  0.9601833333333334 0.9571
#  0.9613833333333334 0.9557
#  0.9658 0.9603
#  0.9673666666666667 0.9607
#  0.9702166666666666 0.9625
#  0.9727166666666667 0.9642
#  0.9734333333333334 0.966
#  0.9752333333333333 0.9677
#  0.97555 0.9676
#  0.9780666666666666 0.9687

#正答率が97%以上に学習できていた。
