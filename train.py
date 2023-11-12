"""Train script."""
import argparse
import os

import cshogi
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

from dataset import BOARD_SIZE, PIECE_KIND_NUM, ShogiDataset
from model import TransformerModel
from training_config import TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    return parser.parse_args()


def generate_board(
    model: TransformerModel, noise_scheduler: DDPMScheduler
) -> cshogi.Board:
    """Generate board from model."""
    x = torch.randn((1, PIECE_KIND_NUM, BOARD_SIZE, BOARD_SIZE)).to("cuda")
    for t in noise_scheduler.timesteps:
        t_tensor = torch.tensor([t]).to("cuda").long()
        with torch.no_grad():
            y = model(x, t_tensor)
            x = noise_scheduler.step(y.cpu(), t_tensor.cpu(), x.cpu()).prev_sample.to(
                "cuda"
            )
    x = torch.argmax(x, dim=1)
    board = cshogi.Board()
    pieces_src = board.pieces
    pieces_in_hand_src = board.pieces_in_hand
    pieces_dst = pieces_src.copy()
    for i in range(9):
        for j in range(9):
            piece = x[0, i, j].item()
            pieces_dst[i * 9 + j] = piece
    board.set_pieces(pieces_dst, pieces_in_hand_src)
    return board


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

    model = TransformerModel(PIECE_KIND_NUM, 6, 384)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=f"{config.output_dir}/logs",
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
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
            # Sample noise to add to the images
            noise = torch.randn(batch.shape).to(batch.device)
            bs = batch.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=batch.device,
            ).long()

            # Add noise
            noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_batch, timesteps)
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
        print(result_str)

        if accelerator.is_main_process:
            is_last = epoch == config.num_epochs - 1
            if (epoch + 1) % config.save_board_epochs == 0 or is_last:
                board = generate_board(model, noise_scheduler)
                svg = board.to_svg()
                with open(f"{config.output_dir}/board_{epoch + 1:04d}.svg", "w") as f:
                    f.write(svg)

            if (epoch + 1) % config.save_model_epochs == 0 or is_last:
                torch.save(
                    model.state_dict(), f"{config.output_dir}/model_{epoch + 1:04d}.pt"
                )
