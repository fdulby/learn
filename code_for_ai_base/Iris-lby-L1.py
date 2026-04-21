import torch
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#load
iris = load_iris()
X_np = iris.data.astype(np.float32)
y_np = iris.target.astype(np.int64)

def stratified_split(X, y, train_ratio=0.4, val_ratio=0.3, seed=42):
    np.random.seed(seed)
    n_classes = len(np.unique(y))
    train_idx, val_idx, test_idx = [], [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        n_total = len(idx)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

train_idx, val_idx, test_idx = stratified_split(X_np, y_np)
X_train_raw, y_train = X_np[train_idx], y_np[train_idx]
X_val_raw, y_val = X_np[val_idx], y_np[val_idx]
X_test_raw, y_test = X_np[test_idx], y_np[test_idx]

scaler = StandardScaler()
X_train = torch.from_numpy(scaler.fit_transform(X_train_raw))
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(scaler.transform(X_val_raw))
y_val = torch.from_numpy(y_val)
X_test = torch.from_numpy(scaler.transform(X_test_raw))
y_test = torch.from_numpy(y_test)

LR = 0.01
L1_LAMBDA = 0
EPOCHS = 300
BATCH_SIZE = 16


class SoftmaxRegression:
    def __init__(self, n_features, n_classes):
        self.W = torch.zeros(n_features, n_classes, requires_grad=True)
        self.b = torch.zeros(n_classes, requires_grad=True)

    def forward(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)

def train(model, X_train, y_train, X_val, y_val, lr, l1_lambda, batch_size, epochs, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_samples = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    marked_epochs = []
    marked_losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            if model.W.grad is not None: model.W.grad.zero_()
            if model.b.grad is not None: model.b.grad.zero_()

            logits = model.forward(X_batch)
            ce_loss = F.cross_entropy(logits, y_batch)
            loss = ce_loss
            if l1_lambda > 0:
                loss = loss + l1_lambda * torch.sum(torch.abs(model.W))

            loss.backward()

            with torch.no_grad():
                model.W -= lr * model.W.grad
                model.b -= lr * model.b.grad

        with torch.no_grad():
            train_logits = model.forward(X_train)
            train_ce = F.cross_entropy(train_logits, y_train)
            train_loss = train_ce + l1_lambda * torch.sum(torch.abs(model.W)) if l1_lambda > 0 else train_ce
            current_train_loss = train_loss.item()
            history['train_loss'].append(current_train_loss)

            val_logits = model.forward(X_val)
            val_loss = F.cross_entropy(val_logits, y_val).item()
            val_pred = model.predict(X_val)
            val_acc = (val_pred == y_val).float().mean().item()
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"[Epoch {epoch + 1:03d}/{epochs}] Train Loss: {current_train_loss:.6f}")
                marked_epochs.append(epoch)
                marked_losses.append(current_train_loss)

    return history, marked_epochs, marked_losses

model = SoftmaxRegression(n_features=4, n_classes=3)
history, marked_epochs, marked_losses = train(
    model, X_train, y_train, X_val, y_val,
    lr=LR, l1_lambda=L1_LAMBDA, batch_size=BATCH_SIZE, epochs=EPOCHS
)

print(f"最终验证集准确率: {history['val_acc'][-1]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
plt.plot(history['val_loss'], label='Val Loss', linewidth=2, color='orange')

plt.scatter(marked_epochs, marked_losses, color='red', s=80, zorder=5, label='Marked Epochs')
for ep, loss in zip(marked_epochs, marked_losses):
    plt.annotate(f'Ep{ep + 1}\n{loss:.3f}',
                 xy=(ep, loss),
                 xytext=(ep + 15, loss + 0.3),
                 fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Curve (lr={LR}, L1_lambda={L1_LAMBDA})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'loss_lr{LR}_l1{L1_LAMBDA}.png', dpi=150)
plt.show()

print("\n========== 模型权重矩阵 W ==========")
print(model.W.detach().numpy())
print("\n偏置 b:")
print(model.b.detach().numpy())

if L1_LAMBDA > 0:
    zero_ratio = (torch.abs(model.W) < 1e-4).float().mean().item()
    print(f"\n权重矩阵中接近0的比例: {zero_ratio:.2%}")

with torch.no_grad():
    y_pred = model.predict(X_test).numpy()

cm = confusion_matrix(y_test.numpy(), y_pred)

class_names = iris.target_names
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap='Blues')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
plt.xticks(range(len(class_names)), class_names)
plt.yticks(range(len(class_names)), class_names)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (lr={LR}, L1_lambda={L1_LAMBDA})')
plt.show()