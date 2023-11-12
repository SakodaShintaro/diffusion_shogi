#!/usr/bin/env python3
import torch
import torch.jit
import torch.nn as nn
from dataset import BOARD_SIZE, PIECE_KIND_NUM

CHANNEL_NUM_TO_NHEAD = {
    256: 8,
    384: 6,
    512: 16,
    768: 12,
    1024: 16,
}

SQUARE_NUM = BOARD_SIZE ** 2


class TransformerModel(nn.Module):
    def __init__(self, input_channel_num: int, block_num: int, channel_num: int) -> None:
        super(TransformerModel, self).__init__()
        nhead = CHANNEL_NUM_TO_NHEAD[channel_num]

        self.first_encoding_ = torch.nn.Linear(input_channel_num, channel_num)
        self.positional_encoding_ = torch.nn.Parameter(
            torch.zeros([SQUARE_NUM, 1, channel_num]), requires_grad=True)
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        channel_num = x.shape[1]
        board_size = x.shape[2]
        x = x.view([batch_size, channel_num, board_size * board_size])
        x = x.permute([2, 0, 1])  # [bs, ch, H*W] -> [H*W, bs, ch]
        x = self.first_encoding_(x)
        x = x + self.positional_encoding_
        x = self.encoder_(x)
        x = self.head_(x)
        x = x.permute([1, 2, 0])  # [H*W, bs, ch] -> [bs, ch, H*W]
        x = x.view([batch_size, channel_num, board_size, board_size])
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
        [batch_size, input_channel_num, BOARD_SIZE, BOARD_SIZE])
    timesteps = torch.randint(0, 1000, (batch_size,)).long()

    out = model(input_data, timesteps)
    print(out.shape)