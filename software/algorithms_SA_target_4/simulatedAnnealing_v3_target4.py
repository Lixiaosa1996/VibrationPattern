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

# 全局变量
data_dict = {}  # 基本振动图像
observation_space = 3773  # 基本图像数据中, comsol中随机取点数目(x,y坐标数目)
judge_plane = {}  # 目标振动图像

# 获取图像坐标范围, 并将坐标插值至179*367
with open(os.path.join('../datasets/patterns_singleMotor', '0_10_-180.pkl'), 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']

grid_x, grid_y = np.mgrid[min(x):max(x):179j, min(y):max(y):367j]  # 比较图像格式 179*367

# 记录模拟退火过程中的相似度变化和对应解法
path_solution = []
path_energy = []

# 振动能量图像插值函数(先求能量再插值)
def interplote_pattern(x, y, z):

    # 根据时序振动数据求能量
    z = np.average(np.sqrt(z ** 2), axis=0) * 1E3  # 计算为振动能量图像

    # 根据比较图像尺寸，对稀疏的振动能量图像进行插值
    points = np.array([x, y, z]).T
    grid_z = griddata(points[:, :2], points[:, 2], (grid_x, grid_y), method='cubic')  # 二维插值为和目标图像一样密集

    # 归一化
    grid_z[np.isnan(grid_z)] = 0
    grid_z -= np.min(grid_z)
    grid_z /= np.max(grid_z)

    return grid_z


# 振动图像相似度评价函数SSIM
def compute_ssim(plane):

    # 对稀疏振动图像求能量图，并插值和归一化
    grid_z = interplote_pattern(x, y, plane)

    # 计算SSIM相似度
    res_ssim = ssim(grid_z, judge_plane, data_range=1)

    return res_ssim


# 能量计算：合成图像与目标图像的SSIM相似度
def energy(solution):
    # Define your energy function here
    init_plane = np.zeros((200, observation_space), dtype=np.float32)

    for i in range(5):
        for j in range(-180, 180):
            init_plane += data_dict[f'{i}_10_{j}']['T'] * solution[i * 360 + j + 180] / 10

    return compute_ssim(init_plane)


# 扩散函数：从当前解随机变化出新解(每次随机变化某个振动马达的单个相位)
def neighbor(solution):
    # Define your neighborhood function here
    new_solution = solution.copy()
    idx = random.randint(0, len(solution) - 1)
    # new_solution[idx] += random.choice([-5, 5])
    # new_solution[idx] += random.randint(-5, 5)
    new_solution[idx] = random.choice(list(range(-10, 0)) + list(range(1, 11)))
    return new_solution


# 模拟退火过程：比较新解和当前解的能量，以一定概率接受较差的新解
def simulated_annealing_core(init_solution, T, alpha, num_iterations):
    current_solution = init_solution
    current_energy = energy(current_solution)

    pbar = tqdm(total=num_iterations)
    for _ in range(num_iterations):

        path_energy.append(current_energy)
        path_solution.append(current_solution)

        new_solution = neighbor(current_solution)
        new_energy = energy(new_solution)

        delta_energy = new_energy - current_energy
        print(_, ':', delta_energy)
        
        # if delta_energy > 0 or random.uniform(0, 1) < math.exp(delta_energy * 1000 / T):
        # 相似度越低，越容易接受较差的新解
        if delta_energy > 0 or random.uniform(0, 1) < math.fabs(1 - new_energy) * T / 1000:
            current_solution, current_energy = new_solution, new_energy
            print(current_energy)

        T *= alpha

        pbar.update()
    pbar.close()

    return current_solution, current_energy


# 模拟退火设置：生成初始解，设定循环次数和退火值
def parallel_sa(num_iterations=1000):
    # Initial solutions for each core
    init_solution = [random.randint(-10, 10) for _ in range(1800)]

    T = 1000
    alpha = 0.999

    best_solution, best_energy = simulated_annealing_core(init_solution, T, alpha, num_iterations)

    # # Find the best solution among all cores
    # best_solution, best_energy = max(results, key=lambda x: x[1])

    return best_solution, best_energy


if __name__ == "__main__":

    # 导入单振动源的震动图像模型
    print('loading basic patterns...')
    for file in tqdm(os.listdir('../datasets/patterns_singleMotor')):
        with open(os.path.join('../datasets/patterns_singleMotor', file), 'rb') as f:
            data = pickle.load(f)
        x, y, T, p = data['x'], data['y'], data['T'][1800:2000].astype(np.float32), data['parameters']
        data_dict[file[:-4]] = {'T': T, 'p': np.array(p, dtype=np.float32)}

    for _ in range(4):

        target_num = _

        path_solution = []
        path_energy = []

        # 导入目标振动图像
        print(f'loading target pattern {target_num}...')
        with open(os.path.join('../datasets/pattern_target_4', f'target_pattern_{target_num}.pkl'), 'rb') as f:
            data = pickle.load(f)
        target_x, target_y, T = data['x'], data['y'], data['T'][1800:2000].astype(np.float32)

        judge_plane = interplote_pattern(target_x, target_y, T)

        # 模拟退火方法
        solution, energy_value = parallel_sa()
        print(f"Best solution {target_num}:", solution)
        print(f"Energy value {target_num}:", energy_value)
        # 打开文件以进行写入最优结果
        with open(f'save_SA_results_4/solution_{target_num}.txt', "w") as f:
            # 将数组转换为字符串并写入到文件的第一行
            array_str = ' '.join(map(str, solution))
            f.write(array_str + '\n')

            # 将浮点数转换为字符串并写入到文件的第二行
            f.write(str(energy_value) + '\n')

            # 将数组的每个元素写入到接下来的每一行
            for j in range(-180, 180, 1):
                for i in range(5):
                    f.write(f"{i * 720 + 180 + j}\t")

                    data = solution[i * 360 + j + 180]
                    f.write(f"{data}\t")

                f.write("\n")

        # 保存energy轨迹
        with open(f'save_SA_results_4/path_energy_{target_num}.txt', 'wb') as f:
            for item in path_energy:
                f.write(("%s\n" % item).encode())

        # 将energy和对应solution都写入到pkl中
        # 保存到pkl文件
        with open(f'save_SA_results_4/path_energy+solution_{target_num}.pkl', 'wb') as f:
            pickle.dump({'path_energy': path_energy, 'path_solution': path_solution}, f)

        # 画出最优图像
        best_plane = np.zeros((200, observation_space), dtype=np.float32)

        for i in range(5):
            for j in range(-180, 180):
                best_plane += data_dict[f'{i}_10_{j}']['T'] * solution[i * 360 + j + 180] / 10

        grid_z = interplote_pattern(x, y, best_plane)

        print('evaluating...')
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.imshow(judge_plane)

        ax2 = fig.add_subplot(122)
        ax2.imshow(grid_z)

        plt.savefig(f'save_SA_results_4/result_compare_100_V{target_num}.png')
        # plt.show()
