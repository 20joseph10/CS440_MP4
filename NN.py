import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix


class NN(object):
    def __init__(self, n_layer=4, n_units=256, learn_rate=0.1):
        train_set = []
        with open('expert_policy.txt') as f:
            content = f.readlines()
        for i in content:
            train_set.append(i.strip().split())

        self.train_data = np.array(train_set).astype('float')
        feature_dim = self.train_data[:, :-1].shape[1]
        self.n_samples = self.train_data.shape[0]
        self.train_data[:, :-1] = (self.train_data[:, :-1] - np.mean(self.train_data[:, :-1], axis=0)[None, :])\
                                  / np.std(self.train_data[:, :-1], axis=0)[None, :]
        self.layer = n_layer
        self.units = n_units
        self.learn_rate = learn_rate

        ws = []
        bs = []
        ws.append(np.random.rand(feature_dim, n_units))
        bs.append(np.zeros(n_units))
        for i in range(1, n_layer-1):
            ws.append(np.random.rand(n_units, n_units))
            bs.append(np.zeros(n_units))
        ws.append(np.random.rand(n_units, 3))
        bs.append(np.zeros(3))

        self.weights = ws
        self.biases = bs

    def train(self, epoch, init_weight):
        for i in range(len(self.weights)):
            self.weights[i] *= init_weight
        self.mini_batch_GD(epoch)

    def test(self, X, y):
        return self.layer_network(X, y, test=True)

    def mini_batch_GD(self, epoch, batch_size=100):
        for e in range(1, epoch+1):
            np.random.shuffle(self.train_data)
            for i in range(0, self.n_samples, batch_size):
                actual_batch = min(batch_size, self.n_samples-i)
                Xy = self.train_data[i:i+actual_batch]
                X = Xy[:, :-1]
                y = Xy[:, -1]
                loss = self.layer_network(X, y, test=False)
            print('Loss in epoch {}: {}'.format(e, loss))

    def layer_network(self, X, y, test):
        acache = []
        rcache = []
        A = X
        for i in range(self.layer-1):
            Z = self.affine_forward(A, i)
            acache.append(A)
            A = self.relu_forward(Z)
            rcache.append(Z)
        F = self.affine_forward(A, self.layer-1)
        acache.append(A)
        if test:
            return np.argmax(F, axis=1)
        loss, dF = self.cross_entropy(F, y)
        dZ = dF
        for i in range(self.layer-1, 0, -1):
            dA, dW, db = self.affine_backward(dZ, acache, i)
            self.update_param(dW, db, i)
            dZ = self.relu_backward(dA, rcache, i-1)
        dX, dW, db = self.affine_backward(dZ, acache, 0)
        self.update_param(dW, db, 0)

        return loss

    def affine_forward(self, A, layer_index):
        return A @ self.weights[layer_index] + self.biases[layer_index][None, :]

    def relu_forward(self, Z):
        A = np.copy(Z)
        A[A < 0] = 0
        return A

    def affine_backward(self, dZ, acache, layer_index):
        dA = dZ @ self.weights[layer_index].T
        dW = acache[layer_index].T @ dZ
        db = np.sum(dZ, axis=0)
        return dA, dW, db

    def relu_backward(self, dA, rcache, layer_index):
        dZ = np.copy(dA)
        dZ[rcache[layer_index] < 0] = 0
        return dZ

    def cross_entropy(self, F, y):
        L1 = np.diag(np.array([F[:, int(i)] for i in y]))
        L2 = logsumexp(F, axis=1)
        L = -np.mean(L1 - L2)

        label_flag = np.zeros(np.shape(F))
        for i in range(len(F)):
            label_flag[i, int(y[i])] = 1
        dF = -(label_flag - np.exp(F) / (np.sum(np.exp(F), axis=1)[:, None])) / len(F)

        return L, dF

    def update_param(self, dW, db, layer_index):
        self.weights[layer_index] -= self.learn_rate * dW
        self.biases[layer_index] -= self.learn_rate * db


nn = NN(n_layer=3)
nn.train(200, 0.1)
# a = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
# print(np.argmax(a, axis=1))
# b = np.array([1.0,2.0,3.0])
# print(np.diag(np.array([a[:, int(i)] for i in b])))
# print(logsumexp(a, axis=1))
# print(a[:,:-1].shape)
# print(np.mean(nn.train_data, axis=0))
# print(np.std(nn.train_data, axis=0))
# print(a-np.std(nn.train_data[:,:-1], axis=0)[None,:])
# for i in range(len(nn.biases)):
#     print(nn.biases[i]+1)
# print(nn.train_data.shape)
# a = np.arange(10)