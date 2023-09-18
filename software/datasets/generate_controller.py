import os

# for i in range(1, 10001, 100):
#     os.system(f'python generate_dataset.py {i}')

for i in range(5):
    for j in range(-180, 180, 10):
        os.system(f'python generate_patterns.py {i} {j}')
