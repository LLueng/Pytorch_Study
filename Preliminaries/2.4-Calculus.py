import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from autograd import grad
import autograd.numpy as np


# 2.4.1. 导数和微分
# 通过令x=1，h接近0，导数为2
def f(x):
    return 3 * x ** 2 - 4 * x
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h
# 数据生成
h_values, derivatives = [], []
true_derivative = 2  # 理论导数值
h = 0.1
# 打印结果：
# h=0.10000, numerical limit=2.30000
# h=0.01000, numerical limit=2.03000
# h=0.00100, numerical limit=2.00300
# h=0.00010, numerical limit=2.00030
# h=0.00001, numerical limit=2.00003
for _ in range(5):
    derivatives.append(numerical_lim(f, 1, h))
    h_values.append(h)
    h *= 0.1
# 绘图设置
plt.figure(figsize=(10, 6))
plt.plot(h_values, derivatives, 'o-', color='blue', label='Numerical Derivative')  # 显式指定颜色
plt.axhline(y=true_derivative, color='red', linestyle='--', linewidth=2, label='True Derivative (2)')
# 坐标轴优化
plt.xscale('log')
plt.gca().invert_xaxis()  # h从大到小排列
plt.xlabel('h (log scale)', fontsize=12)
plt.ylabel('Derivative Value', fontsize=12)
plt.title("Numerical Derivative Convergence at x=1", fontsize=14)
# 样式增强
plt.legend(loc='upper right')  # 强制显示图例
plt.grid(True, which='both', linestyle=':', alpha=0.7)
# 绘图显示
plt.show()

# 2.4.2、偏导数
# 偏导数是针对​​多元函数​​的导数，表示在固定其他变量不变的情况下，函数对某一特定变量的变化率
# 使用 Python SymPy 库
x, y = sp.symbols('x y')
f = x ** 2 + y ** 2
# 计算对x的偏导数
df_dx = sp.diff(f, x)
# 输出：2x
print(f"∂f/∂x = {df_dx}")  
# 计算对y的偏导数
df_dy = sp.diff(f, y)
# 输出：2y
print(f"∂f/∂y = {df_dy}")  

# 2.4.3、梯度
# 梯度是多元函数所有​​偏导数组成的向量​​，指向函数在该点上升最快的方向，模长表示变化率
# 几何意义
# ​​方向​​：函数值增长最快的方向。
# ​​大小​​：函数在该方向的变化速率（例如陡峭程度）
# 使用 NumPy 计算数值梯度
def f(x):
    return x[0] ** 2 + x[1] ** 2
def numerical_gradient(f, x):
    h = 1e-5
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp
    return grad
x = np.array([3.0, 4.0])
grad = numerical_gradient(f, x)
# 输出：[6. 8.]
print(f"梯度向量：{grad}")

# 2.4.4、链式法则
# 链式法则用于计算​​复合函数的导数​​，通过分解复合函数为多个简单函数的组合，逐层求导后相乘
# 应用场景
# ​​反向传播​​：深度学习中计算损失函数对权重的梯度。
# ​​物理模型​​：复杂运动（如弹簧振动）的动力学分析。
# 2.4.4.1、使用 SymPy手动实现
x = sp.symbols('x')
g = 3 * x ** 2 + 5  # 内层函数
f = g ** 4        # 外层函数
# 链式法则计算导数
df_dx = sp.diff(f, x)
# 输出：12*x*(3*x​**​2 + 5)​**​3
print(f"导数：{df_dx}") 
# 2.4.4.2、使用 Autograd自动微分
def composite(x):
    g = 3 * x ** 2 + 5
    return g ** 4 
# 自动计算导数
grad_composite = grad(composite)
x_value = 2.0
# 输出：12 * 2*(3 * 4 +5)^3 = 12 * 2 * 17^3
print(f"x=2时的导数：{grad_composite(x_value)}")  