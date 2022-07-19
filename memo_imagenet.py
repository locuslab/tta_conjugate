import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from pathlib import Path

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import logging
import os

from models import *
from conf import cfg, load_cfg_fom_args

from robustbench.utils import clean_accuracy as accuracy
from robustbench.data import load_cifar10c, load_cifar10, load_cifar100c, load_cifar10, load_imagenetc
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel
import torchvision.models as models 

import tent
import copy

from utils import AugMixDatasetImageNet 
from utils import augmentations

augmentations.IMAGE_SIZE = 224

torch.manual_seed(0)

from tent import copy_model_and_optimizer, load_model_and_optimizer, softmax_entropy

torch.backends.cudnn.enabled=False

from pdb import set_trace as st 

logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_cfg_fom_args('"ImageNet-C evaluation.')
logger.info("test-time adaptation: TENT")

if not os.path.exists(cfg.LOG_DIR):
    os.makedirs(cfg.LOG_DIR)


if cfg.MODEL.ARCH == "resnet50_polyloss":
    net = models.__dict__["resnet50"]().to(device)
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(cfg.MODEL.CKPT_PATH)
    net.load_state_dict(checkpoint["state_dict"])

    class Normalized_Net(nn.Module):
        def __init__(self, net):
            super(Normalized_Net, self).__init__()

            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1).to(device)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1).to(device)
            self.net = net

        def forward(self, x):
            x = (x - self.mu) / self.sigma
            return self.net.forward(x)

    net = Normalized_Net(net)

elif cfg.MODEL.ARCH == "resnet50_pt":
    net = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    net = torch.nn.DataParallel(net)

else:
    pass 

def setup_optimizer(params, lr_test=None):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if lr_test is None:
        lr_test = cfg.OPTIM.LR

    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=lr_test,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=lr_test,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def meta_test_adaptive(model, test_loader, n_inner_iter=1, adaptive=True):
    model = tent.configure_model(model)

    params, _ = tent.collect_params(model)
    inner_opt = setup_optimizer(params)

    if not adaptive:
        model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)

    acc = 0.

    counter = 0 
    num_examples = 0

    counter = 0
    for i, (images, y_curr) in enumerate(test_loader):
        counter += 1
        num_examples += images[0].shape[0]
        y_curr = y_curr.to(device)

        if counter % 50 == 0:
            print("batch id ", counter)

        if not adaptive:
            load_model_and_optimizer(model, inner_opt,
                                 model_state, optimizer_state)

        for _ in range(n_inner_iter):
            T = cfg.OPTIM.TEMP

            logits_aug1 = model(images[1].to(device))
            logits_aug2 = model(images[2].to(device))
            logits_aug3 = model(images[3].to(device))
            p_aug1, p_aug2, p_aug3 = F.softmax(logits_aug1/T, dim=1), F.softmax(logits_aug2/T, dim=1) , F.softmax(logits_aug3/T, dim=1)

            p_avg = (p_aug1 + p_aug2 + p_aug3) / 3
            tta_loss = - (p_avg * torch.log(p_avg)).sum(dim=1)


            tta_loss = tta_loss.mean()

            inner_opt.zero_grad()
            tta_loss.backward()

            inner_opt.step()

        outputs_new = model(images[0].to(device))
        acc += (outputs_new.max(1)[1] == y_curr).float().sum()

    return acc.item() / num_examples

def get_imagenetc_loader(data_dir, corruption, severity, batch_size, shuffle=False):
    data_folder_path = Path(data_dir) / "ImageNet-C"/ corruption / str(severity)

    prepr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.ToTensor()
    ])

    imagenet = datasets.ImageFolder(data_folder_path, prepr)

    preprocess = transforms.Compose(
            [transforms.ToTensor()])
    test_data = AugMixDatasetImageNet(imagenet, preprocess)

    test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=20,
                pin_memory=True)
    return test_loader 

for i, severity in enumerate(cfg.CORRUPTION.SEVERITY):
    err_list = []
    for j, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        test_loader = get_imagenetc_loader("/project_data/datasets", corruption_type, severity, cfg.TEST.BATCH_SIZE, True)

        print("Meta test begin!")
        net_test = copy.deepcopy(net)

        acc = meta_test_adaptive(net_test, test_loader, 1, adaptive=True)

        print("Meta test finish!")
        err = 1. - acc
        err_list.append(err)

        logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
