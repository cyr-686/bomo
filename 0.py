import numpy as np

# ===== 打开 design_top.npz =====
data = np.load('results/design_top.npz')
print('design_top.npz 里的键：', list(data.keys()))

candidates = data['candidates']
pred_target = data['pred_target']
true_target = data['true_target']
true_spectra = data['true_spectra']

print('Top 候选膜厚形状：', candidates.shape)
print('MLP 预测 510 nm 反射率：', pred_target)
print('TMM 验证 510 nm 反射率：', true_target)

print('\nTop 20 候选膜厚：')
print(candidates)