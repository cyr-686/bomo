import numpy as np

def tmm_reflectance(d, wavelengths, n_list, n0=1.0, ns=1.52):
    """
    传输矩阵法计算多层膜反射光谱。
    d: 膜厚数组，长度 4
    wavelengths: 波长数组，长度 41
    n_list: 每层折射率，长度 4
    n0: 入射介质折射率
    ns: 基底折射率
    返回: R，长度与 wavelengths 相同
    """
    R = np.zeros(len(wavelengths), dtype=float)
    for i, lam in enumerate(wavelengths):
        M = np.eye(2, dtype=complex)
        for dj, nj in zip(d, n_list):
            delta = 2.0 * np.pi * nj * dj / lam
            mj = np.array([
                [np.cos(delta), 1j * np.sin(delta) / nj],
                [1j * nj * np.sin(delta), np.cos(delta)]
            ], dtype=complex)
            M = M @ mj

        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]

        num = n0 * (M11 + M12 * ns) - (M21 + M22 * ns)
        den = n0 * (M11 + M12 * ns) + (M21 + M22 * ns)
        r = num / den
        R[i] = np.abs(r) ** 2

    return R


def tmm_batch(d_batch, wavelengths, n_list, n0=1.0, ns=1.52):
    """批量计算反射光谱。"""
    return np.array([
        tmm_reflectance(d, wavelengths, n_list, n0, ns)
        for d in d_batch
    ])