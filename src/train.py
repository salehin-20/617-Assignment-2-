 # src/train.py
import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import GarbageDataset
from model import MultiModalNet


@dataclass
class TrainConfig:
    data_dir: str
    batch_size: int
    epochs: int
    lr: float
    img_size: int
    num_workers: int
    seed: int
    max_text_len: int
    save_path: str


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for images, text_ids, labels in loader:
        images = images.to(device)
        text_ids = text_ids.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images, text_ids)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), labels.detach())
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Folder that contains Black/Blue/Green/Other")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_text_len", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Used only if you don't have a separate val folder")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_text_len=args.max_text_len,
        save_path=args.save_path,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Image transforms
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset (single folder). We'll random-split into train/val.
    full_ds = GarbageDataset(cfg.data_dir, transform=None, max_text_len=cfg.max_text_len)

    n_total = len(full_ds)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    # Different transforms for train/val (wrap by setting attribute)
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )

    model = MultiModalNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "config": vars(args),
                },
                cfg.save_path
            )
            print(f"  ✅ saved best to {cfg.save_path} (val_acc={best_val_acc:.4f})")

    print("Done. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
