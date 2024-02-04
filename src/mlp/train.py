import gc
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics import hr_score, ndcg_score
from src.mlp.data import BookingData
from src.mlp.model import MLPSMF
from src.utils import train_epoch, val_epoch, save_checkpoint, seed_torch, get_device


def train():
    data = BookingData()
    fname = "mlp"

    TRAIN_BATCH_SIZE = 1024
    LR = 1e-3
    EPOCHS = 12
    GRADIENT_ACCUMULATION = 1
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 1024
    DROPOUT_RATE = 0.2
    TRAIN_WITH_TEST = True
    NUM_WORKERS = 8
    NUM_WORKERS = 1

    device = get_device()
    # device = "cpu"
    print("device:", device)

    seed = 0
    seed_torch(seed)

    preds_all = []
    best_scores = []
    best_epochs = []
    for fold in range(5):
        preds_fold = []
        print("#" * 25)
        print(f"### FOLD {fold}")

        train_dataset = data.get_train_dataset(fold, TRAIN_WITH_TEST)
        valid_dataset = data.get_valid_dataset(fold)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=TRAIN_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = MLPSMF(
            data.num_cities + 1,
            data.num_countries + 1,
            data.num_devices,
            5,
            data.low_frequency_city_index,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            dropout=DROPOUT_RATE,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            pct_start=0.1,
            div_factor=1e3,
            max_lr=3e-3,
            epochs=EPOCHS,
            steps_per_epoch=int(np.ceil(len(train_data_loader) / GRADIENT_ACCUMULATION)),
        )
        best_score = 0
        best_epoch = 0

        for epoch in range(EPOCHS):
            print(time.ctime(), "Epoch:", epoch)
            train_loss = train_epoch(train_data_loader, model, optimizer, scheduler, device)
            val_loss, pred, true = val_epoch(valid_data_loader, model, device)
            pred[:, data.low_frequency_city_index] = -1e10  # remove low frequency cities
            hr = hr_score(true, pred)
            ndcg = ndcg_score(true, pred)

            print(
                f"""
            ############# Epoch {epoch} result #############
            # Fold {fold}
            # Seed {seed}
            # Learning Rate: {optimizer.param_groups[0]["lr"]:.7f}
            # Train Loss: {np.mean(train_loss):.4f}
            # Validation Loss: {np.mean(val_loss):.4f}
            # HR Score: {hr:.4f}
            # NDCG Score: {ndcg:.4f}
            #################################################
            """,
                flush=True,
            )
            if ndcg > best_score:
                best_score = ndcg
                best_epoch = epoch
                preds_fold = pred
                save_checkpoint(model, optimizer, scheduler, epoch, best_score, fold, seed, fname)
        del model, scheduler, optimizer, valid_data_loader, valid_dataset, train_data_loader, train_dataset
        gc.collect()

        preds_all.append(preds_fold)
        print(f"fold {fold}, best score: {best_score:.6f} best epoch: {best_epoch:3d}")
        best_scores.append(best_score)
        best_epochs.append(best_epoch)

        print()
        for fold, (best_score, best_epoch) in enumerate(zip(best_scores, best_epochs)):
            print(f"fold {fold}, best score: {best_score:.6f} best epoch: {best_epoch:3d}")
        print(f"seed {seed} best score: {best_score:.6f} best epoch: {np.mean(best_epochs):.1f}")


if __name__ == "__main__":
    train()
