import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import logging 
from models import *
import numpy as np
torch.backends.cudnn.enabled=False
from conf import cfg, load_cfg_fom_args

load_cfg_fom_args('"training')

logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def getloader_SVHN(train, batch_size):
    
    transform = transforms.Compose([
        transforms.RandomCrop([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize([0.4362, 0.4432, 0.4744], [0.1973, 0.2003, 0.1963])
    ])

    if train:
        split = 'train'
    else:
        split = 'test'
    
    dset = datasets.SVHN("./data/", split=split, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=train)
    
    return loader

net = ResNet18()
net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
save_path = cfg.MODEL.SAVE_PATH
train_loader = getloader_SVHN(True, 256)
test_loader = getloader_SVHN(False, 256)

NUM_EPOCHS=50
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
best_acc = 0

ce_loss = nn.CrossEntropyLoss(reduce=False)
smax = torch.nn.Softmax(dim=1)

for epoch in range(NUM_EPOCHS):
    epoch_ce_loss = 0
    for batch_idx, (data,labels) in enumerate(train_loader):

        data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
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
            labels = F.one_hot(labels, 10)*1
            loss = (torch.norm(preds-labels,p=2,dim=1)**2).mean()

        epoch_ce_loss += loss

        loss.backward()
        optimizer.step()
    
    epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
    print("Epoch : {} Loss : {}".format(epoch, epoch_ce_loss))
    scheduler.step()

    acc=0
    total_test_point=0
    for batch_idx, (data,labels) in enumerate(test_loader):

        data, labels = data.cuda(), labels.cuda()
        total_test_point += data.shape[0]
        logits = net(data)
        logits = torch.squeeze(logits, dim = 1)
        
        acc += (logits.max(1)[1] == labels).float().sum()

    acc = acc.item()/ total_test_point
    print("Epoch : {} Acc : {}".format(epoch, acc))

    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), save_path)