import numpy as np

# ===== 个性化参数 =====
STUDENT_LAST2 = 37
STUDENT_LAST6 = 270137

N = STUDENT_LAST2
TARGET_WAVELENGTH = 450 + 10 * (N % 31)   # 510 nm
SEED = STUDENT_LAST6                      # 270137
DESIGN_SEED = SEED + 1                    # 270138

# ===== 统一物理设置 =====
WAVELENGTHS = np.arange(400, 801, 10, dtype=float)  # 41 点
N_H = 2.30
N_L = 1.45
N_S = 1.52
N_AIR = 1.0

LAYER_NAMES = ['H', 'L', 'H', 'L']
LAYER_INDICES = [N_H, N_L, N_H, N_L]

D_MIN = 40.0
D_MAX = 180.0

# ===== 数据集划分 =====
N_TOTAL = 5000
N_TRAIN = 4000
N_VAL = 500
N_TEST = 500

TRAIN_SIZES = [500, 1000, 2000, 4000]

# ===== MLP 训练 =====
MLP_HIDDEN = [128, 128, 64]
BATCH_SIZE = 128
EPOCHS = 2000
LR = 1e-3

TARGET_IDX = int(np.argmin(np.abs(WAVELENGTHS - TARGET_WAVELENGTH)))