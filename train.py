"""Train script."""
import argparse
import datetime
import os

import cshogi
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import random

from dataset import (BOARD_SIZE, HAND_PIECE_KIND_NUM, INPUT_SEQ_LEN,
                     PIECE_KIND_NUM, ShogiDataset)
from model import TransformerModel
from training_config import TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    return parser.parse_args()


def tensor_to_board(x: torch.Tensor) -> cshogi.Board:
    x = torch.argmax(x, dim=1)
    board = cshogi.Board()
    pieces_src = board.pieces
    pieces_in_hand_src = board.pieces_in_hand
    pieces_dst = pieces_src.copy()
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            piece = x[i * BOARD_SIZE + j].item()
            pieces_dst[i * 9 + j] = piece
    for c in range(2):
        for hp in range(7):
            piece = x[BOARD_SIZE * BOARD_SIZE + c * 7 + hp].item()
            pieces_in_hand_src[c][hp] = piece
    turn = x[BOARD_SIZE * BOARD_SIZE + HAND_PIECE_KIND_NUM].item()
    board.set_pieces(pieces_dst, pieces_in_hand_src)
    turn = max(0, min(1, turn))
    board.turn = turn
    return board


def generate_board(
    model: TransformerModel, noise_scheduler: DDPMScheduler, dataset: ShogiDataset
) -> cshogi.Board:
    """Generate board from model."""
    index = random.randrange(0, len(dataset))
    condition, target = dataset[index]
    condition = condition.unsqueeze(0).to("cuda")
    x = torch.randn((1, INPUT_SEQ_LEN, PIECE_KIND_NUM)).to("cuda")
    for t in noise_scheduler.timesteps:
        t_tensor = torch.tensor([t]).to("cuda").long()
        with torch.no_grad():
            y = model(x, t_tensor, condition)
            x = noise_scheduler.step(y.cpu(), t_tensor.cpu(), x.cpu()).prev_sample.to(
                "cuda"
            )
    return tensor_to_board(condition[0]), tensor_to_board(x[0]), tensor_to_board(target)


if __name__ == "__main__":
    args = parse_args()

    config = TrainingConfig()

    dataset = ShogiDataset(args.data_dir)
    print(f"データ数 : {len(dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    model = TransformerModel(PIECE_KIND_NUM, 3, 256)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # Initialize accelerator and tensorboard logging
    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"train_result/{datetime_str}"
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=f"{output_dir}/logs",
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")
        loss_sum = 0
        for batch in train_dataloader:
            condition, target = batch

            # Sample noise to add to the images
            noise = torch.randn(condition.shape).to(condition.device)
            bs = condition.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=condition.device,
            ).long()

            # Add noise
            noisy_batch = noise_scheduler.add_noise(target, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_batch, timesteps, condition)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.update(1)
            accelerator.log(logs, step=global_step)
            loss_sum += loss.detach().item() * bs
            global_step += 1

        loss_avg = loss_sum / len(dataset)

        result_str = ""
        result_str += f"epoch:{epoch + 1:04d}\t"
        result_str += f"global_step:{global_step:08d}\t"
        result_str += f"loss_avg:{loss_avg:.4f}"
        # print(result_str)

        if accelerator.is_main_process:
            is_last = epoch == config.num_epochs - 1
            save_board_epochs = max(1, config.num_epochs // 10)
            if (epoch + 1) % save_board_epochs == 0 or is_last:
                condition, result, target = generate_board(model, noise_scheduler, dataset)
                svg = condition.to_svg()
                with open(f"{output_dir}/condition_{epoch + 1:04d}.svg", "w") as f:
                    f.write(svg)
                svg = result.to_svg()
                with open(f"{output_dir}/board_{epoch + 1:04d}.svg", "w") as f:
                    f.write(svg)
                svg = target.to_svg()
                with open(f"{output_dir}/target_{epoch + 1:04d}.svg", "w") as f:
                    f.write(svg)

            if is_last:
                torch.save(model.state_dict(), f"{output_dir}/model_{epoch + 1:04d}.pt")
