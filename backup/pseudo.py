from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

#? training loop
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()
best_score = 0

for epoch in range(num_epochs):
    loss_total = 0
    model.train()
    for img, gt in train_dataloader:
        img, gt = img.to(device), gt.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred,gt)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    loss_avg = loss_total / len(train_dataloader)
    self.logging(loss_avg)

    if epoch % validate_num == 0:
        val_score = 0
        model.eval()
        with torch.no_grad():
            for img, gt in valid_dataloader:
                img, gt = img.to(device), gt.to(device)
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                val_score += metric(pred,gt).item()
        val_score_average = val_score/ len(valid_dataloader)
        self.logging(val_score_average)
        model.train()
        if val_score_average > best_score:
            best_score = val_score_average
            torch.save(model.state_dict(), 'best_path')
    scheduler.step()

#? test
test_score = 0
model.eval()
with torch.no_grad():
    for img, gt in eval_dataloader:
        img, gt = img.to(device), gt.to(device)
        pre = model(img)
        pred = torch.argmax(pred, dim=1)
        test_score += metric(pre,gt).item()
test_score_avg = test_score/ len(eval_dataloader)
self.logging(test_score_avg)

#? Dataset
class MyDataset(Dataset):
    def __init__(self, path=None, transform=None):
        self.path = path
        self.samples = read(path)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, gt = self.samples[idx]
        if self.transform:
            img, gt = self.transform(img, gt)
        img = torch.from_numpy(img).float()
        gt = torch.tensor(gt).long()
        return img,gt

train_dataset = MyDataset(train_path, train_transform)
valid_dataset = MyDataset(valid_path, valid_transform)
eval_dataset = MyDataset(eval_path, eval_transform)

#? Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4)

#? Model
class UNet(nn.Module):
    def __init__(self, down_channels, up_channels, num_classes):
        super().__init__()
        self.down_blocks = nn.ModuleList(
            [self.makedownblock(down_channels[idx], down_channels[idx+1]) for idx in range(len(down_channels)-1)]
        )
        self.pools = nn.ModuleList(nn.MaxPool2d(kernel_size=2, stride=2) for _ in self.down_blocks)
        self.up_blocks = nn.ModuleList(
            [self.makeupblock(up_channels[idx], up_channels[idx+1]) for idx in range(len(up_channels)-1)]
        )
        self.up_samples = nn.ModuleList(
            [nn.ConvTranspose2d(up_channels[idx], up_channels[idx+1], kernel_size=2, stride=2) for idx in range(len(up_channels)-1)]
            )
        self.bottle_neck = self.makebottleneck(down_channels[-1], up_channels[0])
        self.out_conv = self.makeoutconv(up_channels[-1], num_classes)

    def makedownblock(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        doubleblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
            nn.ReLU(),
        )
        return doubleblock

    def makeupblock(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        doubleblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
            nn.ReLU(),
        )
        return doubleblock
    
    def makebottleneck(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        bottleneck = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
            nn.ReLU(),
        )
        return bottleneck

    def makeoutconv(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        bottleneck = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(64, out_channel, kernel_size=1),
        )
        return bottleneck

    def forward(self, x):
        down_out = []
        up_out = []
        for idx,downblock in enumerate(self.down_blocks):
            x = downblock(x)
            down_out.append(x)
            x = self.pools[idx](x)

        x = self.bottle_neck(x)
        for idx,upblock in enumerate(self.up_blocks):
            x = self.up_samples[idx](x)
            x = torch.cat((x, down_out[-(idx+1)]), dim=1)
            x = upblock(x)
            up_out.append(x)
        
        return self.out_conv(up_out[-1])
