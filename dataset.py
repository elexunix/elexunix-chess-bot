#!/usr/bin/python3
import pandas as pd, numpy as np, torch#, torch.nn as nn, torch.nn.functional as F
import chess
#from tqdm import tqdm, trange
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from torchtune.datasets import ConcatDataset

class ChessDataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, halfKP, i_from=0, i_to=None):
    print("Caricamento dataset...", f'{halfKP=}')
    self.halfKP = halfKP
    df = pd.read_csv(csv_file)
    df = df[i_from:i_to] if i_to is not None else df[i_from:]
    self.n, self.depth = len(df) - 1, df.shape[1] - 6
    self.X_fen = df['fen'].to_numpy()[:n].astype('S100')
    self.y_evals = df[[f'eval_d{d}' for d in range(self.depth+1)]].to_numpy()[:n]
    print(f'calibration info: {np.abs(self.y_evals[:,-1]).mean()=}, {self.y_evals[:,-1].std()=}, {self.y_evals[:,-1][np.abs(self.y_evals[:,-1])<400].std()=}')
    #y_future = df[['eval_5', 'eval_10', 'eval_20']].to_numpy()
    self.y_outcome = df['outcome'].to_numpy()
    print('DATASET at depth', self.depth)
    print(f"Posizioni totali: {self.n}, vittorie bianco {np.sum(self.y_outcome == 1)}, " +
       f"vittorie nero {np.sum(self.y_outcome == -1)}, patte: {np.sum(self.y_outcome == 0)}")
    self.shifts = np.arange(64, dtype=np.uint64)
  def __len__(self):
    return self.n
  def __getitem__(self, i):
    x = np.zeros((772+40960 if self.halfKP else 773,), dtype=np.float32)
    fen = self.X_fen[i].decode('ascii')
    b = chess.Board(fen)
    bitboards = np.array([
      int(b.pieces(piece, color))
      for color in [chess.WHITE, chess.BLACK]
      for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    ], dtype=np.uint64)
    x[:768] = (bitboards[:, None] >> self.shifts & 1).ravel()
    if b.turn is chess.WHITE:
      x[768] = b.has_kingside_castling_rights(chess.WHITE)
      x[769] = b.has_queenside_castling_rights(chess.WHITE)
      x[770] = b.has_kingside_castling_rights(chess.BLACK)
      x[771] = b.has_queenside_castling_rights(chess.BLACK)
      turn_sign = +1
    else:
      x[768] = b.has_kingside_castling_rights(chess.BLACK)
      x[769] = b.has_queenside_castling_rights(chess.BLACK)
      x[770] = b.has_kingside_castling_rights(chess.WHITE)
      x[771] = b.has_queenside_castling_rights(chess.WHITE)
      turn_sign = -1
      x[:768] = x[:768].reshape(2,6,8,8)[::-1,:,::-1,:].flatten()
    if self.halfKP:
      flip = 0 if b.turn is chess.WHITE else 56
      y = np.zeros((64,10,64), dtype=np.float64)
      y[b.king(b.turn)^flip,:5] = x[:320].reshape((5,64))
      y[b.king(b.turn)^flip,5:] = x[384:704].reshape((5,64))
      x[772:] = y.ravel()
    else:
      x[772] = 1 #b.ep_square if b.ep_square is not None else -1
    Lambda = 0.5
    y = Lambda * (2/(1+np.exp(turn_sign * -self.y_evals[i,-1]/400))-1) + (1 - Lambda) * turn_sign * self.y_outcome[i]
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

import bisect
from torch.utils.data import Dataset

class ConcatDataset(Dataset):
  def __init__(self, datasets):
    super(ConcatDataset, self).__init__()
    self.datasets = list(datasets)
    self.cumulative_sizes = self.cumsum(self.datasets)

  @staticmethod
  def cumsum(sequence):
    r = []
    s = 0
    for item in sequence:
      s += len(item)
      r.append(s)
    return r

  def __len__(self):
    return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

  def __getitem__(self, idx):
    if idx < 0:
      if idx + len(self) < 0:
        raise IndexError("Index out of range")
      idx += len(self)

    dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    if dataset_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
    return self.datasets[dataset_idx][sample_idx]
