import torch
from models import *
import torchvision 
import torch.utils.data as data 
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
torch.backends.cudnn.enabled=False
from conf import cfg, load_cfg_fom_args

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_cfg_fom_args('"training')

if cfg.MODEL.DATASET == "cifar10":
    save_path = cfg.MODEL.SAVE_PATH
    net = Normalized_ResNet(depth=26)
    net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

elif cfg.MODEL.DATASET == "cifar100":
    save_path = cfg.MODEL.SAVE_PATH
    net = Normalized_ResNet_CIFAR100()
    net = torch.nn.DataParallel(net)
    net.to(device)
    cudnn.benchmark = True

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

if cfg.MODEL.DATASET == "cifar10":
    train_data = torchvision.datasets.CIFAR10("./data", True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR10("./data", False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
if cfg.MODEL.DATASET == "cifar100":
    train_data = torchvision.datasets.CIFAR100("./data", True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR100("./data", False, transform=transforms.Compose([transforms.ToTensor()]), download=True)

train_loader = data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

ce_loss = nn.CrossEntropyLoss(reduce=False)
smax = torch.nn.Softmax(dim=1)
best_acc = 0.0
for epoch in range(200):
    net.train()
    for batch_idx, (data,labels) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        preds = net(data)
        if cfg.MODEL.LOSS == "cross-entropy":
            loss = ce_loss(preds, labels)
        if cfg.MODEL.LOSS == "polyloss":
            eps = cfg.MODEL.EPS
            P_t = smax(preds)
            P_t_val = P_t[range(data.shape[0]),labels]
            loss = ce_loss(preds, labels) + eps*(1 - P_t_val)
            loss = loss.mean()
        if cfg.MODEL.LOSS == "squared":
            if cfg.MODEL.DATASET == "cifar10":
                labels = F.one_hot(labels, 10)*1
            if cfg.MODEL.DATASET == "cifar100":
                labels = F.one_hot(labels, 100)*1
            loss = (torch.norm(preds-labels,p=2,dim=1)**2).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2.0)

        optimizer.step()

    scheduler.step()
    
    acc=0.0
    
    net.eval()
    for batch_idx, (data,labels) in enumerate(test_loader):
        data, labels = data.cuda(), labels.cuda()
        preds = net(data)
        acc += (preds.max(1)[1] == labels).float().sum()
    
    acc = acc / 10000
    print(f"Epoch : {epoch} : Acc : {acc}")

    if acc > best_acc:
        best_acc = acc 
        torch.save({"net": net.state_dict()}, save_path)