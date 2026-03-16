import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from functions.preprocess import LunaDataset

def main():
    train_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=False,
    )
    val_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=True,
    )

    # 여기서는 학습이 아니라 cache 생성 목적이므로
    # batch_size는 크게 중요하지 않다.
    train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print("Warming up train cache...")
    for _ in tqdm(train_dl, total=len(train_dl)):
        pass

    print("Warming up val cache...")
    for _ in tqdm(val_dl, total=len(val_dl)):
        pass

    print("Cache warmup done.")

if __name__ == "__main__":
    main()