import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect  # noqa


def train_epoch(loader, model, optimizer, scheduler, device):
    model.train()
    train_loss = []

    # with tqdm_logging_redirect():
    for batch in (bar := tqdm(loader)):
        optimizer.zero_grad()

        batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}
        out_dict = model(batch)
        loss = out_dict["loss"]
        loss_np = loss.detach().cpu().numpy()

        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)

        bar.set_description(f"loss: {loss_np:.5f}, smooth loss: {smooth_loss:.5f}")

    return train_loss


def val_epoch(loader, model, device):
    model.eval()
    val_loss = []
    logits = []
    targets = []

    with torch.no_grad():
        # with tqdm_logging_redirect():
        for batch in (bar := tqdm(loader)):
            batch = {k: batch[k].to(device, non_blocking=True) for k in batch.keys()}

            out_dict = model(batch)
            batch_logits = out_dict["logits"]
            loss = out_dict["loss"]
            loss_np = loss.detach().cpu().numpy()
            target = batch["target"]
            logits.append(batch_logits.detach())
            targets.append(target.detach())
            val_loss.append(loss_np)

            smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
            bar.set_description(f"loss: {loss_np:.5f}, smooth loss: {smooth_loss:.5f}")

        val_loss = np.mean(val_loss)

    logits = torch.cat(logits).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    return val_loss, logits, targets


def save_checkpoint(model, optimizer, scheduler, epoch, best_score, fold, seed, fname):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    checkpoint_path = Path(f"./checkpoints/{fname}")
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path / f"{fname}_{fold}_{seed}.pt")


def load_checkpoint(model, fold, seed, fname):
    checkpoint = torch.load(f"./checkpoints/{fname}/{fname}_{fold}_{seed}.pt")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def seed_torch(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
