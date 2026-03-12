from torchvision import datasets, transforms
import sys, os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super().__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.batch_norm.weight, 1.0)
        nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = out + x
        out = torch.relu(out)
        return out


class ResConvNet(nn.Module):
    def __init__(self, n_chans_1=32, n_blocks=10):
        super().__init__()
        self.n_chans_1 = n_chans_1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_chans_1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *[ResBlock(n_chans=n_chans_1) for _ in range(n_blocks)]
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_chans_1,
            out_channels=n_chans_1 // 2,
            kernel_size=3,
            padding=1
        )
        self.fc1 = nn.Linear(4 * 4 * (n_chans_1 // 2), 128)
        self.fc1_dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)   # 32 -> 16
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)                         # 16 -> 8
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2) # 8 -> 4
        out = out.view(-1, 4 * 4 * (self.n_chans_1 // 2))
        out = torch.relu(self.fc1(out))
        out = self.fc1_dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.fc2_dropout(out)
        out = self.fc3(out)
        return out


def make_datasets(data_path):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        data_path, train=True, download=False, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        data_path, train=False, download=False, transform=val_transform
    )
    return train_dataset, val_dataset


def benchmark_num_workers(
    train_dataset,
    val_dataset,
    device,
    worker_candidates=(0, 1, 2, 4, 8),
    batch_size_train=256,
    batch_size_val=64,
    warmup_epochs=1,
    measure_epochs=2,
):
    """
    각 num_workers 후보에 대해
    - 짧은 warmup
    - 실제 train+val 시간 측정
    을 수행하고 평균 epoch 시간을 비교한다.
    """
    results = []

    for nw in worker_candidates:
        print(f"\n[Benchmark] num_workers = {nw}")

        # Windows에서는 num_workers=0일 때 persistent_workers를 쓰면 안 됨
        loader_kwargs = {
            "pin_memory": (device.type == "cuda"),
        }
        if nw > 0:
            loader_kwargs["persistent_workers"] = True

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=nw,
            **loader_kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=nw,
            **loader_kwargs
        )

        model = ResConvNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Warmup
        for _ in range(warmup_epochs):
            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device=device, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)
                    _ = model(imgs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measured epochs
        epoch_times = []

        for epoch in range(measure_epochs):
            t0 = time.perf_counter()

            model.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device=device, non_blocking=True)
                    labels = labels.to(device=device, non_blocking=True)
                    _ = model(imgs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - t0
            epoch_times.append(elapsed)
            print(f"  measured epoch {epoch + 1}: {elapsed:.3f}s")

        avg_time = sum(epoch_times) / len(epoch_times)
        result = {
            "num_workers": nw,
            "avg_epoch_time_sec": avg_time,
            "epoch_times": epoch_times,
        }
        results.append(result)

        print(f"  -> avg epoch time: {avg_time:.3f}s")

        # 메모리 정리
        del model, optimizer, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = sorted(results, key=lambda x: x["avg_epoch_time_sec"])
    return results


def main():
    seed_everything(42)

    BASE_DIR = os.getcwd()
    sys.path.append(BASE_DIR)
    data_path = os.path.join(BASE_DIR, "data", "data-unversioned", "p1ch7")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    train_dataset, val_dataset = make_datasets(data_path)

    # Windows에서는 보통 0, 1, 2, 4 정도부터 보는 걸 추천
    worker_candidates = (0, 1, 2, 4, 6, 8)

    results = benchmark_num_workers(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        worker_candidates=worker_candidates,
        batch_size_train=256,
        batch_size_val=64,
        warmup_epochs=1,
        measure_epochs=2,
    )

    print("\n===== Benchmark Results =====")
    for r in results:
        print(
            f"num_workers={r['num_workers']:>2} | "
            f"avg_epoch_time={r['avg_epoch_time_sec']:.3f}s | "
            f"times={[round(x, 3) for x in r['epoch_times']]}"
        )

    best = results[0]
    print("\n===== Recommended Setting =====")
    print(
        f"Best num_workers = {best['num_workers']} "
        f"(avg epoch time = {best['avg_epoch_time_sec']:.3f}s)"
    )


if __name__ == "__main__":
    main()