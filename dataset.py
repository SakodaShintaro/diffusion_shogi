"""Dataset for shogi."""
from glob import glob

import cshogi
import torch
from cshogi import Parser
from torch.utils.data import Dataset
from tqdm import tqdm

PIECE_KIND_NUM = 31

# 歩, 香, 桂, 銀, 金, 角, 飛
HAND_PIECE_KIND_NUM = 7 * 2

BOARD_SIZE = 9

INPUT_SEQ_LEN = BOARD_SIZE * BOARD_SIZE + HAND_PIECE_KIND_NUM + 1


class ShogiDataset(Dataset[torch.Tensor]):
    """Dataset for shogi."""

    def __init__(self, kifu_dir: str) -> None:
        """Initialize dataset from csa files."""
        kifu_files = glob(f"{kifu_dir}/*.csa")
        self.sfen_list = []
        for kifu_file in tqdm(kifu_files):
            result = Parser.parse_file(kifu_file)
            for one_game in result:
                board = cshogi.Board(one_game.sfen)
                self.sfen_list.append(board.sfen())
                for move in one_game.moves:
                    board.push_csa(cshogi.move_to_csa(move))
                    self.sfen_list.append(board.sfen())

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.sfen_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return position tensor.

        shape = [INPUT_SEQ_LEN, PIECE_KIND_NUM]
        """
        sfen = self.sfen_list[idx]
        board = cshogi.Board(sfen)
        board_tensor = torch.zeros(BOARD_SIZE * BOARD_SIZE, dtype=torch.long)
        for i in range(9):
            for j in range(9):
                piece = board.piece(i * 9 + j)
                board_tensor[i * BOARD_SIZE + j] = piece

        hand_tensor = torch.zeros(HAND_PIECE_KIND_NUM, dtype=torch.long)
        for c in range(2):
            for hp in range(7):
                hand_tensor[c * 7 + hp] = board.pieces_in_hand[c][hp]

        turn_tensor = torch.tensor([board.turn], dtype=torch.long)

        position_tensor = torch.cat([board_tensor, hand_tensor, turn_tensor], dim=0)

        position_tensor = torch.nn.functional.one_hot(position_tensor, PIECE_KIND_NUM)
        position_tensor = position_tensor.to(torch.float32)
        return position_tensor


if __name__ == "__main__":
    dataset = ShogiDataset("./test_data")
    data = dataset[-1]
    print(data)
    print(data.shape)
    x = torch.argmax(data, dim=1)
    print(x)
