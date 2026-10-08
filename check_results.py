import numpy as np

# ===== 打开 dataset.npz =====
data = np.load('results/dataset.npz')
print('dataset.npz 里的键：', list(data.keys()))

d = data['d']
R = data['R']
train_idx = data['train_idx']
val_idx = data['val_idx']
test_idx = data['test_idx']

print('膜厚 d 形状：', d.shape)
print('光谱 R 形状：', R.shape)
print('训练集索引数量：', len(train_idx))
print('验证集索引数量：', len(val_idx))
print('测试集索引数量：', len(test_idx))

print('\n前 5 组膜厚：')
print(d[:5])

print('\n前 5 组光谱（前 5 个波长）：')
print(R[:5, :5])