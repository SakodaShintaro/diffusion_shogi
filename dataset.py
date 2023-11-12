"""Dataset for shogi."""
from glob import glob

import cshogi
import torch
from cshogi import Parser
from torch.utils.data import Dataset
from tqdm import tqdm

PIECE_KIND_NUM = 31
BOARD_SIZE = 9


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
        """Return board tensor."""
        sfen = self.sfen_list[idx]
        board = cshogi.Board(sfen)
        board_tensor = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.long)
        for i in range(9):
            for j in range(9):
                piece = board.piece(i * 9 + j)
                board_tensor[i, j] = piece
        board_tensor = torch.nn.functional.one_hot(board_tensor, PIECE_KIND_NUM)
        board_tensor = board_tensor.to(torch.float)
        board_tensor = board_tensor.permute(2, 0, 1)
        return board_tensor


if __name__ == "__main__":
    dataset = ShogiDataset("./test_data")
    board_tensor = dataset[0]
    print(board_tensor.shape)
    x = torch.argmax(board_tensor, dim=0)
    print(x)
