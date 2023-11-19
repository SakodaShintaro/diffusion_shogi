#!/usr/bin/env python3
import math
import torch
import torch.jit
import torch.nn as nn
from dataset import BOARD_SIZE, PIECE_KIND_NUM, INPUT_SEQ_LEN

CHANNEL_NUM_TO_NHEAD = {
    256: 8,
    384: 6,
    512: 16,
    768: 12,
    1024: 16,
}

SQUARE_NUM = BOARD_SIZE ** 2


class IntEncoding(nn.Module):
    """
    Reference PositionalEncoding in https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_value: int) -> None:
        super().__init__()
        position = torch.arange(max_value).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_value, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            t: Tensor, shape ``[batch_size,]``
        Return:
            embedded, shape``[batch_size, 1, dim]``
        """
        return self.pe[t]  # type: ignore


class TransformerModel(nn.Module):
    def __init__(self, input_channel_num: int, block_num: int, channel_num: int) -> None:
        super(TransformerModel, self).__init__()
        nhead = CHANNEL_NUM_TO_NHEAD[channel_num]

        self.first_encoding_ = torch.nn.Linear(input_channel_num * 2, channel_num)
        self.positional_encoding_ = torch.nn.Parameter(
            torch.zeros([INPUT_SEQ_LEN, 1, channel_num]), requires_grad=True)
        self.step_encoding_ = IntEncoding(channel_num, 1000)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            channel_num,
            nhead=nhead,
            dim_feedforward=channel_num * 4,
            norm_first=True,
            activation="gelu",
            dropout=0.0)
        self.encoder_ = torch.nn.TransformerEncoder(
            encoder_layer, block_num, enable_nested_tensor=False)  # type: ignore
        self.head_ = torch.nn.Linear(channel_num, input_channel_num)

    def encode_position(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute([1, 0, 2])  # [bs, seq, ch] -> [seq, bs, ch]
        x = self.first_encoding_(x)
        x = x + self.positional_encoding_
        return x

    def forward(self, x: torch.Tensor, diffusion_step: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        forward function.

        x.shape              = [batch_size, seq, ch]
        diffusion_step.shape = [batch_size]
        condition.shape      = [batch_size, seq, ch] (ch is the same as x.shape[2])
        """

        # encode position
        x = torch.cat([x, condition], 2)
        x = self.encode_position(x)

        # encode step
        te = self.step_encoding_(diffusion_step)
        te = te.squeeze(1)
        te = te.unsqueeze(0)  # [bs, dim] -> [1, bs, dim]

        # forward
        x = torch.cat([x, te], dim=0)
        x = self.encoder_(x)
        x = x[0:-1]
        x = self.head_(x)
        x = x.permute([1, 0, 2])  # [seq, bs, ch] -> [bs, ch, seq]
        return x


if __name__ == "__main__":
    input_channel_num = PIECE_KIND_NUM
    block_num = 12
    channel_num = 768

    model = TransformerModel(input_channel_num, block_num, channel_num)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    batch_size = 16
    input_data = torch.randn(
        [batch_size, INPUT_SEQ_LEN, input_channel_num])
    timesteps = torch.randint(0, 1000, (batch_size,)).long()

    out = model(input_data, timesteps, input_data)
    print(out.shape)
