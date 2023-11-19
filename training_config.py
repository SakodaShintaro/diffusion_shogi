"""Training configuration."""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""

    train_batch_size = 1024
    num_epochs = 5000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"
