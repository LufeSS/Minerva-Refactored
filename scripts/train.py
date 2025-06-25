import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import typer
from rich.console import Console
from rich.table import Table
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from minerva.data.wikitext import build_dataloader, load_wikitext
from minerva.model import Decoder

console = Console()


class TrainerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Data parameters
    seq_len: int = 128
    batch_size: int = 8
    dataset: str = "wikitext-2-raw-v1"

    # Model parameters
    num_layers: int = 4
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1

    # Training parameters
    epochs: int = 3
    lr: float = 3e-4
    grad_accum: int = 1
    warmup_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"


def shift_labels(inputs: torch.Tensor):
    return inputs[:, :-1].contiguous(), inputs[:, 1:].contiguous()


def main(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a .env file with configuration."
    ),
    seq_len: int = typer.Option(128, help="Sequence length."),
    batch_size: int = typer.Option(8, help="Batch size."),
    epochs: int = typer.Option(3, help="Number of epochs."),
    lr: float = typer.Option(3e-4, help="Learning rate."),
    num_layers: int = typer.Option(4, help="Number of decoder layers."),
    hidden_dim: int = typer.Option(512, help="Hidden dimension."),
    num_heads: int = typer.Option(8, help="Number of attention heads."),
    dropout: float = typer.Option(0.1, help="Dropout rate."),
    grad_accum: int = typer.Option(1, help="Gradient accumulation steps."),
    warmup_ratio: float = typer.Option(0.1, help="Warm-up ratio for scheduler."),
    dataset: str = typer.Option("wikitext-2-raw-v1", help="Dataset name."),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory."),
):
    """Train Minerva on a given dataset."""
    if config_file:
        config = TrainerConfig(_env_file=config_file)
    else:
        # This feels a bit repetitive, there might be a better way to do this with Typer/Pydantic
        config = TrainerConfig(
            seq_len=seq_len,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            grad_accum=grad_accum,
            warmup_ratio=warmup_ratio,
            dataset=dataset,
            checkpoint_dir=checkpoint_dir,
        )

    device = torch.device(config.device)
    console.print(f"Using device: [bold]{device}[/bold]")

    # --- Dataset Loading ---
    console.print(
        f"[bold]Loading dataset[/bold] (seq_len={config.seq_len})…"
    )
    train_ds, vocab_size, _ = load_wikitext(
        "train", block_size=config.seq_len, dataset_variant=config.dataset
    )
    val_ds, _, _ = load_wikitext(
        "validation", block_size=config.seq_len, dataset_variant=config.dataset
    )
    train_loader = build_dataloader(train_ds, config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_ds, config.batch_size, shuffle=False)

    # --- Model Initialization ---
    console.print("Building model…")
    model = Decoder(
        vocab_size=vocab_size,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)

    if torch.cuda.device_count() > 1:
        console.print(
            f"Using {torch.cuda.device_count()} GPUs with DataParallel"
        )
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=config.lr)
    total_update_steps = (len(train_loader) * config.epochs) // config.grad_accum
    warmup_steps = int(total_update_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training Loop ---
    for epoch in range(1, config.epochs + 1):
        # ... training ...
        model.train()
        running_nll = 0.0
        total_tokens = 0
        step_in_epoch = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            inp, tgt = shift_labels(input_ids)

            logits = model(inp)
            ce_loss_sum = F.cross_entropy(
                logits.view(-1, vocab_size), tgt.view(-1), reduction="sum"
            )

            tokens_in_batch = tgt.numel()
            running_nll += ce_loss_sum.item()
            total_tokens += tokens_in_batch

            avg_loss = ce_loss_sum / tokens_in_batch
            (avg_loss / config.grad_accum).backward()

            if (step_in_epoch + 1) % config.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix(loss=f"{avg_loss.item():.3f}")
            step_in_epoch += 1

        avg_train_loss = running_nll / total_tokens
        train_ppl = math.exp(avg_train_loss)

        # ... validation ...
        model.eval()
        val_nll = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                inp, tgt = shift_labels(input_ids)
                logits = model(inp)
                loss_sum = F.cross_entropy(
                    logits.view(-1, vocab_size), tgt.view(-1), reduction="sum"
                )
                val_nll += loss_sum.item()
                val_tokens += tgt.numel()

        avg_val_loss = val_nll / val_tokens
        val_ppl = math.exp(avg_val_loss)

        console.print(
            f"[green]Epoch {epoch}: Train PPL: {train_ppl:.2f}, Val PPL: {val_ppl:.2f}[/green]"
        )

        torch.save(
            model.state_dict(), ckpt_dir / f"decoder_epoch{epoch}.pt"
        )


if __name__ == "__main__":
    typer.run(main) 