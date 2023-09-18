import mph
import numpy as np
import pickle

from matplotlib import pyplot as plt
from scipy.interpolate import griddata

client = mph.start()

# 加载模型
model = client.load('vibration_array_v4.0_target.mph')
client.models()

model.clear()
model.reset()

# 设置全局参数
model.parameter('Ampl_target', '10')

# 打印模型信息
model.parameters()
model.geometries()
model.build()  # 更新几何信息
model.physics()
model.materials()
model.mesh()  # 构建网格

# 求解
model.studies()
model.solve()

# 保存解
(x, y, z, T) = model.evaluate(['x', 'y', 'z', 'w'])

piece_pair = np.where(z == 1)
piece_index = np.where(piece_pair[0] == 2000)
col_index = piece_pair[1][piece_index]

x, y, T = x[0, col_index], y[0, col_index], T[:, col_index]

with open('target_pattern.pkl', 'wb') as f:
    pickle.dump({'x': x, 'y': y, 'T': T}, f)

# 读取数据，并画图
with open('target_pattern.pkl', 'rb') as f:
    data = pickle.load(f)
x, y, T = data['x'], data['y'], data['T'].astype(np.float32)

# 画出能量图
grid_x, grid_y = np.mgrid[min(x):max(x):179j, min(y):max(y):367j]
z = np.average(np.sqrt(T[1800:2000] ** 2), axis=0) * 1E9
points = np.array([x, y, z]).T
grid_z = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='cubic')
grid_z[np.isnan(grid_z)] = 0

fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(111)
ax1.imshow(grid_z)

plt.savefig(f'target_pattern.png')
plt.show()