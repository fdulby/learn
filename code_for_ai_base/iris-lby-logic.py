import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 数据加载与预处理
iris = load_iris()

# IRIS feature names:
feature_names = iris.feature_names

# IRIS feature values (cm) in numpy ndarray:
data = iris.data

# IRIS species names
species_names = iris.target_names

# IRIS species ids (0='setosa', 1='versicolor', 2='virginica'):
species = iris.target

# Training and test data
X = data
y = (species == 0).astype(int)  # 转为二分类：setosa(0)为1，其它为0

X_trn = X[0::2]  # training data
X_tst = X[1::2]  # test data
y_trn = y[0::2]  # training class label
y_tst = y[1::2]  # test class label

# 为了处理bias，在特征矩阵中增加一列全为 1 的常数
X_trn_bias = np.c_[np.ones(X_trn.shape[0]), X_trn]
X_tst_bias = np.c_[np.ones(X_tst.shape[0]), X_tst]

# 逻辑回归

def sigmoid(z):
    """Sigmoid """
    return 1 / (1 + np.exp(-z))


def compute_loss(X, y, theta):
    """交叉熵损失"""
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-15  # 免log(0)
    loss = -1 / m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return loss


def gradient_descent(X, y, X_val, y_val, lr, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    train_loss_history = []
    val_loss_history = []

    for _ in range(epochs):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y))
        theta -= lr * gradient

        train_loss_history.append(compute_loss(X, y, theta))
        val_loss_history.append(compute_loss(X_val, y_val, theta))

    return train_loss_history, val_loss_history


# 学习率影响

learning_rates = [0.1, 0.01, 0.001]
epochs = 1000

plt.figure(figsize=(16, 5))

for i, lr in enumerate(learning_rates):
    trn_loss, tst_loss = gradient_descent(X_trn_bias, y_trn, X_tst_bias, y_tst, lr, epochs)

    plt.subplot(1, 3, i + 1)
    plt.plot(trn_loss, label='Train Loss', color='blue')
    plt.plot(tst_loss, label='Test Loss', color='red', linestyle='--')
    plt.title(f'Learning Rate: {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()