from torchvision import datasets, transforms
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time



class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
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
    def __init__(self, n_chans_1 = 32, n_blocks=10):
        super().__init__()
        self.n_chans_1 = n_chans_1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_chans_1, kernel_size=3, padding=1)
        # self.resblocks = nn.Sequential(
        #     *(n_blocks * [ResBlock(n_chans = n_chans_1)]))
        self.resblocks = nn.Sequential(
            *[ResBlock(n_chans=n_chans_1) for _ in range(n_blocks)]
        )
        self.conv2 = nn.Conv2d(in_channels=n_chans_1, out_channels=n_chans_1 // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * (n_chans_1 // 2), 128)
        self.fc1_dropout = nn.Dropout(p = 0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_dropout = nn.Dropout(p = 0.2)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x:torch.Tensor):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = out.view(-1, 4 * 4 * (self.n_chans_1 // 2))
        out = torch.relu(self.fc1(out))
        out = self.fc1_dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.fc2_dropout(out)
        out = self.fc3(out)
        return out






def training_loop(n_epochs, optimizer:torch.optim.AdamW, model:ResConvNet, loss_fn:nn.CrossEntropyLoss,
                  scheduler, train_loader, val_loader, device, data_path):
    train_loss_list = []
    val_loss_list = []
    val_loss_curr = float("inf")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, val_loss = 0.0, 0.0
        
        model.train()
        for imgs, labels in train_loader:
            imgs: torch.Tensor
            labels: torch.Tensor
            imgs = imgs.to(device = device, non_blocking=True)
            labels = labels.to(device = device, non_blocking=True)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs: torch.Tensor
                labels: torch.Tensor
                imgs = imgs.to(device = device, non_blocking=True)
                labels = labels.to(device = device, non_blocking=True)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)            
                
                val_loss += loss.item()
                
        train_loss_list.append(train_loss / len(train_loader))
        val_loss_list.append(val_loss / len(val_loader))
        current_lr = optimizer.param_groups[0]["lr"]    

        if (val_loss / len(val_loader)) <= val_loss_curr:
            save_path = os.path.join(data_path, 'res_convnet_augmented_best.pt')
            torch.save(model.state_dict(), save_path)
            val_loss_curr = val_loss / len(val_loader)
            print("New best saved")             
            
        if epoch <= 10 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Train loss : {train_loss / len(train_loader):.4f}, "
                  f"Validation loss : {val_loss / len(val_loader):.4f}, "
                  f"lr: {current_lr:.6f}, "
                  f"elapsed {time.time() - t0 :.4f}s")
            
        scheduler.step() 
            
    
    return train_loss_list, val_loss_list










def main():
    BASE_DIR = os.getcwd()
    sys.path.append(BASE_DIR)
    data_path = os.path.join(BASE_DIR, "data", "data-unversioned", "p1ch7")


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
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device : {device}")
    
    NUM_WORKERS = 4
    loader_kwargs = {
        "pin_memory": (device.type == "cuda"),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    cifar10 = datasets.CIFAR10(
        data_path, train=True, download=False, transform=train_transform
    )

    cifar10_val = datasets.CIFAR10(
        data_path, train=False, download=False, transform=val_transform
    )





    # train_loader = torch.utils.data.DataLoader(cifar2, batch_size = 256, shuffle=True,
    #                                         num_workers=4, pin_memory=True, persistent_workers=True)
    # val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size = 64, shuffle=False,
    #                                         num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size = 256, shuffle=True, num_workers=NUM_WORKERS, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size = 64, shuffle=False, num_workers=NUM_WORKERS, **loader_kwargs)

    model = ResConvNet()
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
    loss_fn = nn.CrossEntropyLoss()


    train_loss_list, val_loss_list = training_loop(
        n_epochs=400,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device = device,
        data_path = data_path
    )

    save_path = os.path.join(data_path, 'res_convnet_augmented.pt')
    torch.save(model.state_dict(), save_path)

    train_loss_list_np, val_loss_list_np = [], []

    for i in range(len(train_loss_list)):
        x, y = train_loss_list[i], val_loss_list[i]
        train_loss_list_np.append(x)
        val_loss_list_np.append(y)
        
        

    import matplotlib.pyplot as plt

    plt.plot(train_loss_list_np, color = 'lightblue', label = 'Train Loss')
    plt.plot(val_loss_list_np, color = 'orange', label = 'Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    main()