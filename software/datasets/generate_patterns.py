import mph
import sys
import torch
import pickle
import numpy as np

client = mph.start()
model = client.load('vibration_array_v4.0.mph')


def model_evaluate(model, parameters):
    model.clear()
    model.reset()
    model.parameter('Ampl1', f'{parameters[0]}')
    model.parameter('freq1', '160')
    model.parameter('off1', f'{parameters[1]}')
    model.parameter('Ampl2', f'{parameters[2]}')
    model.parameter('freq2', '160')
    model.parameter('off2', f'{parameters[3]}')
    model.parameter('Ampl3', f'{parameters[4]}')
    model.parameter('freq3', '160')
    model.parameter('off3', f'{parameters[5]}')
    model.parameter('Ampl4', f'{parameters[6]}')
    model.parameter('freq4', '160')
    model.parameter('off4', f'{parameters[7]}')
    model.parameter('Ampl5', f'{parameters[8]}')
    model.parameter('freq5', '160')
    model.parameter('off5', f'{parameters[9]}')
    model.parameter('period_num', '20')

    model.build()
    model.mesh()
    model.solve()
    (x, y, z, T) = model.evaluate(['x', 'y', 'z', 'w'])

    piece_pair = np.where(z == 1)
    piece_index = np.where(piece_pair[0] == 2000)
    col_index = piece_pair[1][piece_index]

    return x[0, col_index], y[0, col_index], T[:, col_index]

# 使用generate_controller.py进行生成，参数为电机编号+相位
motor_index = 0 if len(sys.argv) < 3 else int(sys.argv[1])
phase = 0 if len(sys.argv) < 3 else int(sys.argv[2])
if __name__ == '__main__':
    for i in range(phase, phase + 10):
        parameters = []
        
        #生成给每个电机的幅值和相位设置
        for j in range(5):
            if motor_index == j:
                parameters.extend([10, i * np.pi / 180])
            else:
                parameters.extend([0, 0])
        print(parameters)
        
        x, y, T = model_evaluate(model, parameters)
        # torch.save({'x': x, 'y': y, 'T': T, 'parameters': parameters}, f'patterns_singleMotor/{motor_index}_10_{i}.pth')
        with open(f'patterns-singleMotor/{motor_index}_10_{i}.pkl', 'wb') as f:
            pickle.dump({'x': x, 'y': y, 'T': T, 'parameters': parameters}, f)

