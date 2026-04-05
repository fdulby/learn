import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



#data prepare
iris = load_iris()
X = iris.data      # 特征数据
y = iris.target    # 类别标签

# 类别索引: 0='setosa', 1='versicolor', 2='virginica'
# 特征索引: 0=sepal length, 1=sepal width, 2=petal length, 3=petal width

print("problem-1")

# setosa (0) 和 versicolor (1)
mask_q1 = (y == 0) | (y == 1)
data_q1 = X[mask_q1]

# 计算 sepal length (特征 0) 的均值和方差
sepal_length_mean = np.mean(data_q1[:, 0])
sepal_length_var = np.var(data_q1[:, 0])

# 计算 sepal width (特征 1) 的均值和方差
sepal_width_mean = np.mean(data_q1[:, 1])
sepal_width_var = np.var(data_q1[:, 1])

print(f"sepal length 均值: {sepal_length_mean:.2f}")
print(f"sepal length 方差: {sepal_length_var:.2f}")
print(f"sepal width 均值: {sepal_width_mean:.2f}")
print(f"sepal width 方差: {sepal_width_var:.2f}\n")


print("problem-2")

# virginica (2)
mask_q2 = (y == 2)
data_q2 = X[mask_q2]

# 计算 petal length (特征 2) 的均值和方差
petal_length_mean = np.mean(data_q2[:, 2])
petal_length_var = np.var(data_q2[:, 2])

# 计算 petal width (特征 3) 的均值和方差
petal_width_mean = np.mean(data_q2[:, 3])
petal_width_var = np.var(data_q2[:, 3])

print(f"petal length 均值: {petal_length_mean:.2f}")
print(f"petal length 方差: {petal_length_var:.2f}")
print(f"petal width 均值: {petal_width_mean:.2f}")
print(f"petal width 方差: {petal_width_var:.2f}\n")


#MSE当作loss
print("problem-3")

# train
train_mask = (y == 0) | (y == 1)
# X_train 需要是二维数组，因此使用 .reshape(-1, 1)
X_train = X[train_mask, 1].reshape(-1, 1)  # 自变量: sepal width
y_train = X[train_mask, 0]                 # 因变量: sepal length

# test
test_mask = (y == 2)
X_test = X[test_mask, 1].reshape(-1, 1)    # 自变量: sepal width
y_test = X[test_mask, 0]                   # 因变量: sepal length

model = LinearRegression()
model.fit(X_train, y_train)

#参数
coef = model.coef_[0]
intercept = model.intercept_
print(f"(Weight/Coef): {coef:.4f}")
print(f"(Intercept): {intercept:.4f}")
print(f" sepal_length = {coef:.4f} * sepal_width + {intercept:.4f}")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n模型在测试集上的均方误差 (MSE): {mse:.4f}")