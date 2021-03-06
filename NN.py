import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


class NN(object):
    def __init__(self, n_layer=4, n_units=256, learn_rate=0.1, model_dir=None):
        train_set = []
        ws = []
        bs = []

        if model_dir is None:
            with open('expert_policy.txt') as f:
                content = f.readlines()
            for i in content:
                train_set.append(i.strip().split())

            self.train_data = np.array(train_set).astype('float')
            feature_dim = self.train_data[:, :-1].shape[1]
            self.n_samples = self.train_data.shape[0]
            self.sample_mean = np.mean(self.train_data[:, :-1], axis=0)
            self.sample_std = np.std(self.train_data[:, :-1], axis=0)
            self.train_data[:, :-1] = (self.train_data[:, :-1] - np.mean(self.train_data[:, :-1], axis=0)[None, :]) \
                                      / np.std(self.train_data[:, :-1], axis=0)[None, :]
            self.units = n_units
            self.learn_rate = learn_rate

            ws.append(np.random.rand(feature_dim, n_units))
            bs.append(np.zeros(n_units))
            for i in range(1, n_layer - 1):
                ws.append(np.random.rand(n_units, n_units))
                bs.append(np.zeros(n_units))
            ws.append(np.random.rand(n_units, 3))
            bs.append(np.zeros(3))

        else:
            load_data = [None] * 2
            file = glob.glob(model_dir + '/sample_mean.npy')
            self.sample_mean = np.load(file[0])
            file = glob.glob(model_dir + '/sample_std.npy')
            self.sample_std = np.load(file[0])
            for file in glob.glob(model_dir + '/*.npy'):
                load_data[:] = file.strip().split(sep='.')[1].split(sep='_')
                load_data[0] = load_data[0].split(sep='\\')[1]
                if load_data[0] == 'W':
                    ws.append(np.load(model_dir + '/W_' + load_data[1] + '.npy'))
                if load_data[0] == 'B':
                    bs.append(np.load(model_dir + '/B_' + load_data[1] + '.npy'))

        self.weights = ws
        self.biases = bs
        self.layer = len(self.weights)

    def train(self, epoch, init_weight):
        for i in range(len(self.weights)):
            self.weights[i] *= init_weight
        loss_curve, acc_curve = self.mini_batch_GD(epoch)

        os.makedirs(os.path.dirname('./data and plots/'), exist_ok=True)
        np.save('./data and plots/sample_mean.npy', self.sample_mean)
        np.save('./data and plots/sample_std.npy', self.sample_std)
        file_prefix = ['./data and plots/W_', './data and plots/B_']
        for i in range(len(self.weights)):
            save_file_name = file_prefix[0] + str(i)
            np.save(save_file_name, self.weights[i])
            save_file_name = file_prefix[1] + str(i)
            np.save(save_file_name, self.biases[i])
        print('train finished, model file written')

        # figure
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(211)
        ax1.plot(loss_curve)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2 = fig.add_subplot(212)
        ax2.plot(acc_curve)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        fig.suptitle('Loss & Accuracy vs. Epochs in NN with ' +
                     str(self.layer) + ' layers, ' + str(self.units) + ' neurons\n' +
                     'learning rate=' + str(self.learn_rate) +
                     ', initial weight scale=' + str(init_weight),
                     fontsize=20)
        plt.savefig('./data and plots/train_curve.png')

    def test(self, X):
        return self.layer_network(X)

    def mini_batch_GD(self, epoch, batch_size=100):
        loss_epoch = np.zeros(epoch)
        acc_epoch = np.zeros(epoch)
        for e in range(1, epoch + 1):
            np.random.shuffle(self.train_data)
            for i in range(0, self.n_samples, batch_size):
                actual_batch = min(batch_size, self.n_samples - i)
                Xy = self.train_data[i:i + actual_batch]
                X = Xy[:, :-1]
                y = Xy[:, -1]
                loss = self.layer_network(X, y, test=False)
            loss_epoch[e - 1] = loss
            pred = self.test(self.train_data[:, :-1])
            acc_epoch[e - 1] = accuracy_score(self.train_data[:, -1], pred)
            print('Loss in epoch {}: {}'.format(e, loss))
        return loss_epoch, acc_epoch

    def layer_network(self, X, y=None, test=True):
        acache = []
        rcache = []
        A = X
        for i in range(self.layer - 1):
            Z = self.affine_forward(A, i)
            acache.append(A)
            A = self.relu_forward(Z)
            rcache.append(Z)
        F = self.affine_forward(A, self.layer - 1)
        acache.append(A)
        if test:
            return np.argmax(F, axis=1)
        loss, dF = self.cross_entropy(F, y)
        dZ = dF
        for i in range(self.layer - 1, 0, -1):
            dA, dW, db = self.affine_backward(dZ, acache, i)
            self.update_param(dW, db, i)
            dZ = self.relu_backward(dA, rcache, i - 1)
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
        dF = -(label_flag - np.exp(F - L2[:, None])) / len(F)

        return L, dF

    def update_param(self, dW, db, layer_index):
        self.weights[layer_index] -= self.learn_rate * dW
        self.biases[layer_index] -= self.learn_rate * db


if __name__ == '__main__':

    # if we want to read model, uncomment next line, comment all following lines
    # nn = NN(model_dir='./data and plots')

    nn = NN(n_layer=4, n_units=256, learn_rate=0.1)
    nn.train(epoch=500, init_weight=0.025)

    Y = nn.train_data[:, -1]
    prediction = nn.test(nn.train_data[:, :-1])
    print('\nClassification Error: {}'.format(1-accuracy_score(Y, prediction)))
    df_cm = pd.DataFrame(confusion_matrix(Y, prediction), index=[i for i in ['UP', 'NONE', 'DOWN']],
                         columns=[i for i in ['UP', 'NONE', 'DOWN']])
    plt.figure(figsize=(10, 8))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig('./data and plots/conf_mat.png')
