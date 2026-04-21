import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#load
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

def stratified_split(X, y, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3, seed=0):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(X, y, seed=0)

def standardize_fit(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # 防止除以0
    return mean, std

def standardize_transform(X, mean, std):
    return (X - mean) / std

X_mean, X_std = standardize_fit(X_train)
X_train_std = standardize_transform(X_train, X_mean, X_std)
X_val_std = standardize_transform(X_val, X_mean, X_std)
X_test_std = standardize_transform(X_test, X_mean, X_std)

def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y

Y_train = y_train
Y_val = y_val
Y_test = y_test

lr = 0.05
lambda_reg = 1e-3


class SoftmaxRegressionL1:
    def __init__(self, n_features, n_classes, lr=0.05, reg_lambda=1e-3, epochs=300, batch_size=16, seed=42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.W = None
        self.b = None
        self.history = None

    def _init_params(self):
        rng = np.random.default_rng(self.seed)
        self.W = 0.01 * rng.standard_normal((self.n_features, self.n_classes))
        self.b = np.zeros(self.n_classes)

    def _softmax(self, logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, X):
        logits = X @ self.W + self.b
        probs = self._softmax(logits)
        return probs

    def compute_loss(self, X, y):
        probs = self.forward(X)
        n = X.shape[0]

        ce_loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        l1_loss = self.reg_lambda * np.sum(np.abs(self.W))
        total_loss = ce_loss + l1_loss

        return total_loss, ce_loss, l1_loss

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)  #

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        self._init_params()

        n_samples = X_train.shape[0]
        Y_train = y_train

        self.history = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_l1_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                batch_size_actual = xb.shape[0]

                #forward
                logits = xb @ self.W + self.b
                probs = self._softmax(logits)

                #交叉上
                grad_logits = (probs - yb) / batch_size_actual
                grad_W = xb.T @ grad_logits
                grad_b = np.sum(grad_logits, axis=0)

                # L1 次梯度（不对偏置做正则）
                grad_W += self.reg_lambda * np.sign(self.W)

                # 参数更新
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

            train_loss, train_ce, train_l1 = self.compute_loss(X_train, y_train)
            train_acc = self.accuracy(X_train, y_train)

            self.history["train_loss"].append(train_loss)
            self.history["train_ce_loss"].append(train_ce)
            self.history["train_l1_loss"].append(train_l1)
            self.history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss, _, _ = self.compute_loss(X_val, y_val)
                val_acc = self.accuracy(X_val, y_val)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                print(f"Epoch {epoch+1:3d} | "
                      f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

#train
model = SoftmaxRegressionL1(
    n_features=X_train_std.shape[1],
    n_classes=3,
    lr=lr,
    reg_lambda=lambda_reg,
    epochs=500,
    batch_size=16,
    seed=0
)

model.fit(X_train_std, y_train, X_val_std, y_val, verbose=True)

#loss曲线
plt.plot(model.history["train_loss"], label="train loss")
plt.plot(model.history["val_loss"], label="val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Loss Curve (lr={lr}, λ={lambda_reg})")
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

y_test_pred = model.predict(X_test_std)
cm = confusion_matrix(y_test, y_test_pred)

plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(np.arange(3), class_names)
plt.yticks(np.arange(3), class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.show()