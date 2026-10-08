import os
import numpy as np
from config import *
from tmm import tmm_batch


def generate_dataset(save_path='results/dataset.npz'):
    os.makedirs('results', exist_ok=True)

    # 用 seed 生成 5000 组膜厚
    np.random.seed(SEED)
    d_all = np.random.uniform(D_MIN, D_MAX, size=(N_TOTAL, 4))

    # TMM 生成反射光谱
    R_all = tmm_batch(d_all, WAVELENGTHS, LAYER_INDICES, N_AIR, N_S)

    # 固定划分 4000 / 500 / 500
    idx = np.arange(N_TOTAL)
    np.random.seed(SEED)
    np.random.shuffle(idx)

    train_idx = idx[:N_TRAIN]
    val_idx = idx[N_TRAIN:N_TRAIN + N_VAL]
    test_idx = idx[N_TRAIN + N_VAL:]

    np.savez(
        save_path,
        d=d_all,
        R=R_all,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )

    print(f'Dataset saved to {save_path}')
    print(f'target wavelength = {TARGET_WAVELENGTH} nm')
    print(f'seed = {SEED}, design_seed = {DESIGN_SEED}')

    return d_all, R_all, train_idx, val_idx, test_idx


if __name__ == '__main__':
    generate_dataset()