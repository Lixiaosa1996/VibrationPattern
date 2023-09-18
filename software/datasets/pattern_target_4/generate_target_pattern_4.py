import mph
import numpy as np
import pickle

from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from create_new_cmap import new_cmap


# 设置4个目标振动生成点
points_x = [10, 20, -25, 10]
points_y = [25, -25,  -10,  0]

client = mph.start()

# 加载模型
model = client.load('vibration_array_v4.0_target_4.mph')

for _ in range(1):

    i = _

    model.clear()
    model.reset()

    # 设置全局参数
    model.parameter('Ampl_target', '10')
    model.parameter('target_x', f'{points_x[i]}')
    model.parameter('target_y', f'{points_y[i]}')

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

    with open(f'target_pattern_{i}.pkl', 'wb') as f:
        pickle.dump({'x': x, 'y': y, 'T': T}, f)

    # 读取数据，并画图
    with open(f'target_pattern_{i}.pkl', 'rb') as f:
        data = pickle.load(f)
    x, y, T = data['x'], data['y'], data['T'].astype(np.float32)

    # 画出能量图
    grid_x, grid_y = np.mgrid[min(x):max(x):179j, min(y):max(y):367j]
    z = np.average(np.sqrt(T[1800:2000] ** 2), axis=0) * 1E9
    points = np.array([x, y, z]).T
    grid_z = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='cubic')
    grid_z[np.isnan(grid_z)] = 0
    grid_z -= np.min(grid_z)
    grid_z /= np.max(grid_z)

    plt.axis('off')  # 不显示坐标轴
    plt.imshow(grid_z, cmap=new_cmap)
    plt.savefig(f'energy_{i}.png', bbox_inches='tight', pad_inches=0)

