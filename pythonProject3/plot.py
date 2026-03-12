import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 读取 .mat 文件
data = scipy.io.loadmat('data.mat')

# 假设数据中包含变量 'beta_distances1', 'beta_distances2', 'beta_distances3', 'elapsed_time1', 'elapsed_time2', 'elapsed_time3'
beta_distances1 = data['beta_distances1'].flatten()  # 假设是一个向量，使用 .flatten() 转为一维数组
beta_distances2 = data['beta_distances2'].flatten()
beta_distances3 = data['beta_distances3'].flatten()
elapsed_time1 = data['elapsed_time1'].flatten()
elapsed_time2 = data['elapsed_time2'].flatten()
elapsed_time3 = data['elapsed_time3'].flatten()

# 如果 max_iter 不在数据中，假设为一个常数
# 这里我们假设 max_iter 是标量，如果不是，请根据实际情况进行修改
max_iter = 1000000

# 检查数据是否有相同的长度
assert len(beta_distances1) == len(beta_distances2) == len(beta_distances3), "beta_distances 的数组长度不一致"
assert len(elapsed_time1) == len(elapsed_time2) == len(elapsed_time3), "elapsed_time 的数组长度不一致"

# 创建绘图
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 绘制三种算法在迭代次数上的比较
ax[0].plot(np.arange(1, len(beta_distances1) + 1), beta_distances1, '-o', label='算法1', linewidth=1.5)
ax[0].plot(np.arange(1, len(beta_distances2) + 1), beta_distances2, '-o', label='算法2', linewidth=1.5)
ax[0].plot(np.arange(1, len(beta_distances3) + 1), beta_distances3, '-o', label='算法3', linewidth=1.5)

ax[0].set_xlabel("迭代次数")
ax[0].set_ylabel("与最优解的距离")
ax[0].set_title("三种算法在迭代次数上的表现")
ax[0].legend(loc="best")
ax[0].grid(True)

# 绘制三种算法在运行时间上的比较
ax[1].plot(elapsed_time1, beta_distances1, '-o', label='算法1', linewidth=1.5)
ax[1].plot(elapsed_time2, beta_distances2, '-o', label='算法2', linewidth=1.5)
ax[1].plot(elapsed_time3, beta_distances3, '-o', label='算法3', linewidth=1.5)

ax[1].set_xlabel("运行时间 (秒)")
ax[1].set_ylabel("与最优解的距离")
ax[1].set_title("三种算法在运行时间上的表现")
ax[1].legend(loc="best")
ax[1].grid(True)

# 显示图形
plt.tight_layout()
plt.show()