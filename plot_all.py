import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    WAVELENGTHS, TARGET_WAVELENGTH, TARGET_IDX,
    SEED, DESIGN_SEED
)
from model import MLP

os.makedirs('results', exist_ok=True)

# =========================================================
# Figure 4: 训练/验证损失曲线
# 需要先按后面第 3 节修改 train_mlp.py 并重新训练，
# 才会生成 train_loss_size_4000.npy 和 val_loss_size_4000.npy
# =========================================================
def plot_loss_curve():
    train_path = 'results/train_loss_size_4000.npy'
    val_path = 'results/val_loss_size_4000.npy'

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print('跳过 Figure 4：未找到 loss 文件，请先按第 3 节修改并重新运行 train_mlp.py')
        return

    train_loss = np.load(train_path)
    val_loss = np.load(val_path)

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label='Train MSE')
    plt.plot(val_loss, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title('Training and validation loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/fig4_loss_curve.png', dpi=300)
    plt.close()
    print('已生成 results/fig4_loss_curve.png')


# =========================================================
# Figure 5: 至少 3 个测试样本的 TMM vs MLP 光谱对比
# =========================================================
def plot_test_samples():
    data = np.load('results/dataset.npz')
    d_all, R_all = data['d'], data['R']
    test_idx = data['test_idx']

    test_d = d_all[test_idx]
    test_R = R_all[test_idx]

    model = MLP()
    model.load_state_dict(torch.load('results/model_size_4000.pt'))
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(test_d, dtype=torch.float32)).numpy()

    # 选 3 个误差中等的样本，展示更直观
    mse_per_sample = ((pred - test_R) ** 2).mean(axis=1)
    order = np.argsort(mse_per_sample)
    picks = [order[len(order)//4], order[len(order)//2], order[3*len(order)//4]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, i in zip(axes, picks):
        ax.plot(WAVELENGTHS, test_R[i], 'o-', label='TMM', markersize=3)
        ax.plot(WAVELENGTHS, pred[i], 's--', label='MLP', markersize=3)
        ax.axvline(TARGET_WAVELENGTH, color='gray', linestyle=':', label='Target')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title(f'Test sample {i}, MSE={mse_per_sample[i]:.4f}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('results/fig5_test_samples.png', dpi=300)
    plt.close()
    print('已生成 results/fig5_test_samples.png')


# =========================================================
# Figure 6: 数据量影响
# =========================================================
def plot_data_size():
    res = np.load('results/data_size_results.npy')
    # 列：size, val_mse, val_mae, test_mse, test_mae
    sizes = res[:, 0]
    val_mse = res[:, 1]
    test_mse = res[:, 3]

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, val_mse, 'o-', label='Validation MSE')
    plt.plot(sizes, test_mse, 's-', label='Test MSE')
    plt.xlabel('Training set size')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title('Effect of training-set size on test error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/fig6_data_size_vs_mse.png', dpi=300)
    plt.close()
    print('已生成 results/fig6_data_size_vs_mse.png')


# =========================================================
# Figure 7: MLP 筛选 vs TMM 验证
# =========================================================
def plot_design():
    data = np.load('results/design_top.npz')
    pred = data['pred_target']
    true = data['true_target']

    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, c='tab:blue')
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    plt.plot(lims, lims, 'r--', label='y = x')
    plt.xlabel(f'TMM reflectance at {TARGET_WAVELENGTH:.0f} nm')
    plt.ylabel(f'MLP predicted reflectance at {TARGET_WAVELENGTH:.0f} nm')
    plt.title('MLP vs TMM for top candidates')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/fig7_mlp_vs_tmm.png', dpi=300)
    plt.close()
    print('已生成 results/fig7_mlp_vs_tmm.png')


# =========================================================
# Figure 8: 失败案例
# =========================================================
def plot_failure():
    data = np.load('results/dataset.npz')
    d_all, R_all = data['d'], data['R']
    test_idx = data['test_idx']

    test_d = d_all[test_idx]
    test_R = R_all[test_idx]

    model = MLP()
    model.load_state_dict(torch.load('results/model_size_4000.pt'))
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(test_d, dtype=torch.float32)).numpy()

    mse_per_sample = ((pred - test_R) ** 2).mean(axis=1)
    worst = int(np.argmax(mse_per_sample))

    plt.figure(figsize=(6, 4))
    plt.plot(WAVELENGTHS, test_R[worst], 'o-', label='TMM', markersize=3)
    plt.plot(WAVELENGTHS, pred[worst], 's--', label='MLP', markersize=3)
    plt.axvline(TARGET_WAVELENGTH, color='gray', linestyle=':', label='Target')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'Worst test sample, MSE={mse_per_sample[worst]:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/fig8_failure_case.png', dpi=300)
    plt.close()
    print(f'已生成 results/fig8_failure_case.png，最差样本膜厚={test_d[worst]}')


if __name__ == '__main__':
    plot_loss_curve()
    plot_test_samples()
    plot_data_size()
    plot_design()
    plot_failure()
    print('全部绘图完成，请查看 results/ 文件夹。')