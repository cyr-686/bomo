import os
import numpy as np
import torch

from config import *
from model import MLP
from tmm import tmm_batch


def main():
    os.makedirs('results', exist_ok=True)

    # 用 design_seed 生成 10000 组新候选
    np.random.seed(DESIGN_SEED)
    candidates = np.random.uniform(D_MIN, D_MAX, size=(10000, 4))

    # 加载 4000 组训练得到的 MLP
    model = MLP()
    model.load_state_dict(torch.load('results/model_size_4000.pt'))
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(candidates, dtype=torch.float32)).numpy()

    target_idx = TARGET_IDX
    # 这里假设目标为：在 target 波长处反射率最大化
    # 如果老师要求最小化或接近某值，修改此处即可
    scores = pred[:, target_idx]

    top_idx = np.argsort(scores)[::-1][:20]
    top_candidates = candidates[top_idx]
    top_pred = scores[top_idx]

    # 用 TMM 重新验证
    top_true_spectra = tmm_batch(
        top_candidates,
        WAVELENGTHS,
        LAYER_INDICES,
        N_AIR,
        N_S
    )
    top_true_target = top_true_spectra[:, target_idx]

    np.savez(
        'results/design_top.npz',
        candidates=top_candidates,
        pred_target=top_pred,
        true_target=top_true_target,
        true_spectra=top_true_spectra
    )

    print(f'Target wavelength: {TARGET_WAVELENGTH} nm, index: {target_idx}')
    print('Top 20 candidates:')
    for i in range(len(top_candidates)):
        print(f'{i+1:2d}: d={top_candidates[i]}, '
              f'MLP={top_pred[i]:.6f}, TMM={top_true_target[i]:.6f}')


if __name__ == '__main__':
    main()