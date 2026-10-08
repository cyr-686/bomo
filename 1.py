import numpy as np

res = np.load('results/data_size_results.npy')
print('size, val_mse, val_mae, test_mse, test_mae')
print(res)

data = np.load('results/dataset.npz')
d_all, R_all = data['d'], data['R']
test_idx = data['test_idx']

import torch
from model import MLP

model = MLP()
model.load_state_dict(torch.load('results/model_size_4000.pt'))
model.eval()

test_d = d_all[test_idx]
test_R = R_all[test_idx]

with torch.no_grad():
    pred = model(torch.tensor(test_d, dtype=torch.float32)).numpy()

mse_per_sample = ((pred - test_R) ** 2).mean(axis=1)
worst = int(np.argmax(mse_per_sample))
print('最差测试样本膜厚:', test_d[worst])
print('最差测试样本 MSE:', mse_per_sample[worst])
