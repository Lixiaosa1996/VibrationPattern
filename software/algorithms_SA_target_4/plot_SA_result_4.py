import os
import pickle
import matplotlib.pyplot as plt
import random

import math
import multiprocessing
import random

import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim

from create_new_cmap import new_cmap


# 导入单振动源的震动图像模型
data_dict = {}
print('loading data...')
for file in tqdm(os.listdir('../datasets/patterns_singleMotor')):
    with open(os.path.join('../datasets/patterns_singleMotor', file), 'rb') as f:
        data = pickle.load(f)
    x, y, T, p = data['x'], data['y'], data['T'][1800:2000].astype(np.float32), data['parameters']
    data_dict[file[:-4]] = {'T': T, 'p': np.array(p, dtype=np.float32)}


for _ in range(4):

    test_num = _

    # 画出目标图像

    with open(f'../datasets/pattern_target_4/target_pattern_{test_num}.pkl', 'rb') as f:
        data = pickle.load(f)
    x, y, T = data['x'], data['y'], data['T'].astype(np.float32)

    observation_size = (179, 367)  # 目标图像尺寸179*367
    grid_x, grid_y = np.mgrid[min(x):max(x):179j, min(y):max(y):367j]
    # init_plane = np.zeros((200, observation_space), dtype=np.float32)
    init_plane = T[1800:2000]

    z = np.average(np.sqrt(init_plane ** 2), axis=0) * 1E3  # 计算为振动能量图像
    points = np.array([x, y, z]).T
    grid_z = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='cubic')  # 二维插值为和目标图像一样密集
    grid_z[np.isnan(grid_z)] = 0
    grid_z -= np.min(grid_z)
    grid_z /= np.max(grid_z)

    plt.axis('off')  # 不显示坐标轴
    plt.imshow(grid_z, cmap=new_cmap)
    plt.savefig(f'energy_result_4/target_{test_num}.png', bbox_inches='tight', pad_inches=0)


    # 画出生成图像
    observation_size = (179, 367)  # 目标图像尺寸179*367
    grid_x, grid_y = np.mgrid[min(x):max(x):179j, min(y):max(y):367j]
    # 从pkl文件中读取
    with open(f'save_SA_results_4/path_energy+solution_{test_num}.pkl', 'rb') as f:
        data = pickle.load(f)
        path_energy = data['path_energy']
        path_solution = data['path_solution']

    energy = path_energy[999] # 最大相似度解
    print(energy)

    # 合成振动图像
    solution = path_solution[999]
    observation_space = 3773 # comsol中随机取点数目(x,y坐标数目)
    init_plane = np.zeros((200, observation_space), dtype=np.float32)
    for i in range(5):
        for j in range(-180, 180):
            init_plane += data_dict[f'{i}_10_{j}']['T'] * solution[i * 360 + j + 180] / 10

    with open(os.path.join('../datasets/patterns_singleMotor', '0_10_-180.pkl'), 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y'] # 获取x和y
    z = np.average(np.sqrt(init_plane ** 2), axis=0) * 1E3  # 计算为振动能量图像
    points = np.array([x, y, z]).T
    grid_z = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='cubic')  # 二维插值为和目标图像一样密集
    grid_z[np.isnan(grid_z)] = 0
    grid_z -= np.min(grid_z)
    grid_z /= np.max(grid_z)

    plt.axis('off')  # 不显示坐标轴
    plt.imshow(grid_z, cmap=new_cmap)
    plt.savefig(f'energy_result_4/composite_{test_num}.png', bbox_inches='tight', pad_inches=0)




    # # 创建折线图
    # plt.plot(path_energy)
    # plt.title('List1 Data Points')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.show()
