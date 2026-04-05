#这里最大的改动就是加入了数据中心化
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

# IRIS feature names:
feature_names = iris.feature_names

# IRIS feature values (cm) in numpy ndarray:
data = iris.data

# IRIS species names
species_names = iris.target_names

# IRIS species ids (0='setosa', 1='versicolor', 2='virginica'):
species = iris.target

# You can start your code here：

# 问题 1：setosa + versicolor 类的 sepal 特征统计
print("问题 1 数据统计结果")

#  setosa (0) 和 versicolor (1)
mask_q1 = (species == 0) | (species == 1)
data_q1 = data[mask_q1]

# 特征索引：sepal length 是 0, sepal width 是 1
sepal_length_mean = np.mean(data_q1[:, 0])
sepal_length_var = np.var(data_q1[:, 0])

sepal_width_mean = np.mean(data_q1[:, 1])
sepal_width_var = np.var(data_q1[:, 1])

print(f"sepal length 的均值: {sepal_length_mean:.2f}")
print(f"sepal length 的方差: {sepal_length_var:.2f}")
print(f"sepal width 的均值: {sepal_width_mean:.2f}")
print(f"sepal width 的方差: {sepal_width_var:.2f}\n")


# 问题 2：virginica 类的 petal 特征统计
print("问题 2 数据统计结果")

# virginica (2)
mask_q2 = (species == 2)
data_q2 = data[mask_q2]

# 特征索引：petal length 是 2, petal width 是 3
petal_length_mean = np.mean(data_q2[:, 2])
petal_length_var = np.var(data_q2[:, 2])

petal_width_mean = np.mean(data_q2[:, 3])
petal_width_var = np.var(data_q2[:, 3])

print(f"petal length 的均值: {petal_length_mean:.2f}")
print(f"petal length 的方差: {petal_length_var:.2f}")
print(f"petal width 的均值: {petal_width_mean:.2f}")
print(f"petal width 的方差: {petal_width_var:.2f}\n")


# 问题 3：构建线性回归模型
print("问题 3 线性回归模型")

# 数据预处理
train_mask = (species == 0) | (species == 1)
x_train = data[train_mask, 1]
y_train = data[train_mask, 0]

test_mask = (species == 2)
x_test = data[test_mask, 1]
y_test = data[test_mask, 0]


# 在这里进行数据中心化
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)
x_train = x_train - x_mean  # 覆盖原变量，使地形变成正圆
y_train = y_train - y_mean  # 覆盖原变量

# 梯度下降
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 500
N = len(x_train)

# grad train
loss_history = []  # 用于记录每轮的 Loss 值

for i in range(epochs):
    # 前向
    y_pred = w * x_train + b
    current_loss = np.mean((y_pred - y_train) ** 2)
    loss_history.append(current_loss)

    # grad
    # MSE = (1/N) * sum((w*x + b - y)^2)
    dw = (2 / N) * np.sum(x_train * (y_pred - y_train))
    db = (2 / N) * np.sum(y_pred - y_train)

    # 更新
    w = w - learning_rate * dw
    b = b - learning_rate * db

# 将偏置 b 还原回真实的截距
b = b + y_mean - w * x_mean


# 导出并保存图片
import os
import matplotlib.pyplot as plt
save_dir = "/Users/liubingyi/Downloads/iris"
#检查是否存在·
os.makedirs(save_dir, exist_ok=True)
filename = f"loss-ep={epochs}-lr={learning_rate}-centring.png"
save_path = os.path.join(save_dir, filename)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), loss_history, color='tab:blue', label=f'LR={learning_rate}')
plt.title(f'Training Loss (Epochs={epochs}, LR={learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig(save_path, bbox_inches='tight')
print(f"Loss 曲线图: {save_path}")
plt.show()

# 模型数据
print("finish!")
print(f" w (斜率): {w:.6f}")
print(f" b (截距): {b:.6f}")
print(f"function:sepal_length = {w:.4f} * sepal_width + {b:.4f}\n")

# predict
y_pred_test = w * x_test + b
mse_test = np.mean((y_test - y_pred_test) ** 2)

print(f"virginica上的 MSE: {mse_test:.4f}")