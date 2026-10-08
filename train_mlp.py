import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import *
from model import MLP


def train_model(train_d, train_R, val_d, val_R,
                seed=SEED, epochs=EPOCHS,
                batch_size=BATCH_SIZE, lr=LR):
    torch.manual_seed(seed)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_d_t = torch.tensor(train_d, dtype=torch.float32)
    train_R_t = torch.tensor(train_R, dtype=torch.float32)
    val_d_t = torch.tensor(val_d, dtype=torch.float32)
    val_R_t = torch.tensor(val_R, dtype=torch.float32)

    n = len(train_d_t)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = train_d_t[idx]
            y = train_R_t[idx]

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_d_t)
                val_loss = criterion(val_pred, val_R_t).item()
            print(f'  epoch {epoch+1}/{epochs}, val MSE = {val_loss:.6f}')

    model.eval()
    with torch.no_grad():
        val_pred = model(val_d_t).numpy()

    val_mse = mean_squared_error(val_R_t.numpy(), val_pred)
    val_mae = mean_absolute_error(val_R_t.numpy(), val_pred)

    return model, val_mse, val_mae


def main():
    os.makedirs('results', exist_ok=True)

    data = np.load('results/dataset.npz')
    d_all, R_all = data['d'], data['R']
    train_idx, val_idx, test_idx = data['train_idx'], data['val_idx'], data['test_idx']

    train_pool_d = d_all[train_idx]
    train_pool_R = R_all[train_idx]

    val_d = d_all[val_idx]
    val_R = R_all[val_idx]

    test_d = d_all[test_idx]
    test_R = R_all[test_idx]

    results = []

    for size in TRAIN_SIZES:
        print(f'Training with {size} samples ...')
        model, val_mse, val_mae = train_model(
            train_pool_d[:size],
            train_pool_R[:size],
            val_d,
            val_R,
            seed=SEED
        )

        model.eval()
        with torch.no_grad():
            test_pred = model(torch.tensor(test_d, dtype=torch.float32)).numpy()

        test_mse = mean_squared_error(test_R, test_pred)
        test_mae = mean_absolute_error(test_R, test_pred)

        results.append([size, val_mse, val_mae, test_mse, test_mae])
        torch.save(model.state_dict(), f'results/model_size_{size}.pt')

        print(f'  val MSE={val_mse:.6f}, val MAE={val_mae:.6f}, '
              f'test MSE={test_mse:.6f}, test MAE={test_mae:.6f}')

    np.save('results/data_size_results.npy', np.array(results))
    print('Data size study results saved.')


if __name__ == '__main__':
    main()