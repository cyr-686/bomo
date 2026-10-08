import numpy as np
import torch
import torch.nn as nn
from config import *
from model import MLP

data = np.load('results/dataset.npz')
d_all, R_all = data['d'], data['R']
train_idx, val_idx = data['train_idx'], data['val_idx']

train_d = d_all[train_idx]
train_R = R_all[train_idx]
val_d = d_all[val_idx]
val_R = R_all[val_idx]

torch.manual_seed(SEED)
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_d_t = torch.tensor(train_d, dtype=torch.float32)
train_R_t = torch.tensor(train_R, dtype=torch.float32)
val_d_t = torch.tensor(val_d, dtype=torch.float32)
val_R_t = torch.tensor(val_R, dtype=torch.float32)

n = len(train_d_t)
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(n)
    epoch_loss = 0.0
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        pred = model(train_d_t[idx])
        loss = criterion(pred, train_R_t[idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= n
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(val_d_t), val_R_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 500 == 0:
        print(f'epoch {epoch+1}, train={epoch_loss:.6f}, val={val_loss:.6f}')

np.save('results/train_loss_size_4000.npy', np.array(train_losses))
np.save('results/val_loss_size_4000.npy', np.array(val_losses))
torch.save(model.state_dict(), 'results/model_size_4000.pt')
print('loss 文件已保存')