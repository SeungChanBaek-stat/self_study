from torchvision import datasets, transforms
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)
data_path = os.path.join(BASE_DIR, "data", "data-unversioned", "p1ch7")


cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ]))

cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ]))



class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

label_map = {i:i for i in range(10)}
cifar = [(img, label_map[label]) for img, label in cifar10 if label in [i for i in range(10)]] # 리스트 컴프리헨션을 통한 데이터 재구축 및 레이블 매핑
cifar_val = [(img, label_map[label]) for img, label in cifar10_val if label in [i for i in range(10)]] # 되게 깔끔하다. (이 방법 말고도 다양한 방법 가능)





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

loaded_model = ResConvNet()
numel_list = [p.numel() for p in loaded_model.parameters()]
print(f"number of params : {sum(numel_list), numel_list}")


load_path = os.path.join(data_path, 'res_convnet_augmented.pt')
loaded_model.load_state_dict(torch.load(load_path))



train_loader = torch.utils.data.DataLoader(cifar, batch_size = 64, shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar_val, batch_size = 64, shuffle=False)

def validate(model, train_loader, val_loader):
    model.eval()
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim = 1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
                
        print(f"Accuracy {name} : {correct / total :.4f}")
        
validate(loaded_model, train_loader, val_loader)