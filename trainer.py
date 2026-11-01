#!/usr/bin/python3
import modin.pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import chess
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from dataset import ChessDataset, StreamingChessDataset, ConcatDataset

#folder = 'datasets/'
folder = '/media/elexunix/hugedrive/datasets/elexunix-chess-bot/'
halfKP = False#True
# dataset07_d6 = ChessDataset(folder+'dataset07_d6.csv', halfKP)
# dataset06_d8 = ChessDataset(folder+'dataset06_d8_processed.csv', halfKP)
# dataset07_d8 = ChessDataset(folder+'dataset07_d8_processed.csv', halfKP)
# dataset08_d8_1 = ChessDataset(folder+'dataset08_d8_1_processed.csv', halfKP)
# dataset08_d8_2 = ChessDataset(folder+'dataset08_d8_2_processed.csv', halfKP)
# dataset09_d8_0 = ChessDataset(folder+'dataset09_d8_0_processed.csv', halfKP)
# dataset09_d8_1 = ChessDataset(folder+'dataset09_d8_1_processed.csv', halfKP)
# dataset09_d8_2 = ChessDataset(folder+'dataset09_d8_2_processed.csv',halfKP)

train_filename = folder+'dataset11_d4_0_train.csv'
test_filename = folder+'dataset11_d4_0_test_small.csv'
dataset_last_train = StreamingChessDataset(train_filename, halfKP, 4)
dataset_last_test = StreamingChessDataset(test_filename, halfKP, 4)
# dataset_last = ChessDataset(last_filename, halfKP)
# n = len(dataset_last)
# n_train = int(.8 * n)
# dataset_last_train = ChessDataset(last_filename, halfKP, i_to=n_train)
# dataset_last_test = ChessDataset(last_filename, halfKP, i_from=n_train, i_to=n)
# together_train_1 = ConcatDataset([dataset06_d8, dataset07_d8,
#                                   dataset08_d8_1, dataset08_d8_2, dataset09_d8_0, dataset09_d8_1,
# ])
together_train_1 = None
together_train_2 = dataset_last_train
together_test = dataset_last_test
print(f'train {(len(together_train_1 or [])+len(together_train_2))/1e6}M, test {len(together_test)/1e6}M')

s1, s2, s3 = 128, 16, 16
device = 'cuda'
nn_model = nn.Sequential(
    #nn.EmbeddingBag(num_embeddings=41734, embedding_dim=s1, mode='sum', padding_idx=41733),  # 41732:bias
                                            # thankfully, padding_idx's grads are explicitly zeroed by torch
    nn.Linear(in_features=773, out_features=s1),
    nn.ReLU(),
    nn.Linear(in_features=s1, out_features=s2),
    nn.ReLU(),
    nn.Linear(in_features=s2, out_features=s3),
    nn.ReLU(),
    nn.Linear(in_features=s3, out_features=1)
).to(device)
nn_model.load_state_dict(torch.load('models/model10m07.pth'))

#nn_model_s = nn.Sequential(
#    nn.Linear(in_features=773, out_features=s1),
#    nn.ReLU(),
#    nn.Linear(in_features=s1, out_features=8),
#    nn.ReLU(),
#    nn.Linear(in_features=8, out_features=8),
#    nn.ReLU(),
#    nn.Linear(in_features=8, out_features=1)
#).to(device)
#nn_model_s.load_state_dict(torch.load('models/model10h01.pth'))
#with torch.no_grad():
#    ch, ch_s = list(nn_model.children()), list(nn_model_s.children())
#    ch[0].weight.zero_()
#    ch[0].weight[:772,:] = ch_s[0].weight[:,:772].T
#    ch[0].weight[41732] = ch_s[0].bias + ch_s[0].weight[:,772]
#    for l in [2, 4, 6]:
#        ch[l].weight.copy_(ch_s[l].weight)
#        ch[l].bias.copy_(ch_s[l].bias)

B = 16384
train_dataloader_1 = None#DataLoader(together_train_1, B, True, num_workers=24, pin_memory=True, drop_last=False)
train_dataloader_2 = DataLoader(together_train_2, B, False, num_workers=24, pin_memory=True, drop_last=False)
# train_dataloader = DataLoader(together_train, B, True, num_workers=24, pin_memory=True, drop_last=False)
test_dataloader = DataLoader(together_test, B, False, num_workers=24, pin_memory=True, drop_last=False)

def save_model(model, name):
    torch.save(nn_model.state_dict(), 'models/'+name+'.pth')
    state = nn_model.state_dict()
    def get_w(name, transpose=False):
        return np.ascontiguousarray(state[name].cpu().numpy().T, dtype=np.float32)
    if halfKP:
      w01 = np.ascontiguousarray(state['0.weight'][:41732].cpu().numpy(), dtype=np.float32)
      b1 = np.ascontiguousarray(state['0.weight'][41732].cpu().numpy(), dtype=np.float32)
    else:
      w01 = get_w('0.weight')
      b1 = get_w('0.bias', True)
    w12 = get_w('2.weight')
    b2  = get_w('2.bias', True)
    w23 = get_w('4.weight')
    b3  = get_w('4.bias', True)
    w34 = get_w('6.weight')
    b4  = get_w('6.bias', True)
    with open('models/'+name+'.bin', 'wb') as f:
        w01.tofile(f)
        b1.tofile(f)
        w12.tofile(f)
        b2.tofile(f)
        w23.tofile(f)
        b3.tofile(f)
        w34.tofile(f)
        b4.tofile(f)
    w01.shape, b1.shape
    print(f'model {name}.bin/pth saved')

#save_model(nn_model,f'model10h_pre')

optimizer = torch.optim.AdamW(nn_model.parameters(), 1e-4, weight_decay=0)
ep_start, ep_end = 18, 28
gamma = 1 - 0/(ep_end-ep_start) / (len(train_dataloader_1 or []) + len(train_dataloader_2 or []))  # e²≈7.4, e³≈20, e⁴≈55
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

losses = []
nn_model.eval()
#with torch.no_grad():
#  for X, y in tqdm(test_dataloader):
#    y_batch_pred = nn_model(X.to(device))
#    y_batch_true = y.to(device).view(-1, 1)
#    loss = F.mse_loss(y_batch_pred, y_batch_true)
#    losses.append(loss.item())
#print(f'before training test loss {np.mean(losses)}')

epoch_train_losses, epoch_test_losses, best_test_loss = [], [], 1e9
for epoch_i in range(ep_start,ep_end):
  nn_model.train()
  nn_model.to(device)
  for train_dataloader_i, train_dataloader_ in enumerate([train_dataloader_2]):
    losses = []
    bar = tqdm(train_dataloader_)
    for i, (X, y) in enumerate(bar):
      y_batch_true = y.to(device).view(-1, 1)
      y_batch_pred = nn_model(X.to(device))  #X[:,:768+5]
      optimizer.zero_grad()
      loss = F.mse_loss(y_batch_pred, y_batch_true)
      loss.backward()
      losses.append(loss.item())
      optimizer.step()
      lr_scheduler.step()
      if i % 10 == 0:
        bar.set_description(f'loss={np.mean(losses[-100:]):.4f}, lr={lr_scheduler.get_last_lr()[0]:2f}')
#       if i % 2000 == 0:
#         save_model(nn_model,f'model10f1_tmp')
    epoch_train_losses.append(np.mean(losses))
    print(f'epoch {epoch_i}-{train_dataloader_i} train loss {epoch_train_losses[-1]}')
    losses = []
    nn_model.eval()
    with torch.no_grad():
      for X, y in tqdm(test_dataloader):
        y_batch_pred = nn_model(X.to(device))
        y_batch_true = y.to(device).view(-1, 1)
        loss = F.mse_loss(y_batch_pred, y_batch_true)
        losses.append(loss.item())
    epoch_test_losses.append(np.mean(losses))
    print(f'epoch {epoch_i} test loss {epoch_test_losses[-1]}')
    if True or epoch_test_losses[-1] < best_test_loss:
      best_test_loss = epoch_test_losses[-1]
      save_model(nn_model,f'model10m{epoch_i:02}')

#nn_model(torch.tensor(X[0,:768]).view(1,12,8,8))
#plt.plot(np.mean(np.array(losses[:len(losses)//1000*1000]).reshape((-1,1000)), 1))
# plt.plot(epoch_train_losses)
# plt.plot(epoch_test_losses)
# plt.show()
