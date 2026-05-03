"""
Bi-LSTM IO model.

Input:  (B, T=100, C=6) normalized IMU window
Output: delta_p (B, 3), delta_R_6d (B, 6)
"""

import torch
import torch.nn as nn


class BiLSTM_IO(nn.Module):
    def __init__(self,
                 input_size: int = 6,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 head_hidden: int = 128,
                 head_dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )

        # Bi-LSTM output dim = 2 * hidden_size
        feat_dim = 2 * hidden_size

        self.trans_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.PReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 3),
        )

        self.rot_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.PReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 6),
        )

        self._init_lstm_biases()

    def _init_lstm_biases(self):
        """Initialize LSTM forget gate bias to 1.0.

        nn.LSTM stores biases as a single vector of size 4*hidden_size per
        direction per layer, with order [i, f, g, o]. We set the 'f' chunk to 1.
        """
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                # Forget gate slice: indices [n//4 : n//2]
                param.data.fill_(0.0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, imu):
        """
        imu: (B, T, 6)
        Returns:
            delta_p:    (B, 3)
            delta_R_6d: (B, 6)
        """
        out, _ = self.lstm(imu)          # (B, T, 2*hidden)
        feat = out.mean(dim=1)           # (B, 2*hidden)
        delta_p = self.trans_head(feat)
        delta_R_6d = self.rot_head(feat)
        return delta_p, delta_R_6d
