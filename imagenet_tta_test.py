from ast import Pass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models 

from pathlib import Path

import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import logging
import os
import tqdm

import numpy as np

from models import *
from conf import cfg, load_cfg_fom_args

from robustbench.utils import clean_accuracy as accuracy
from robustbench.data import load_cifar10c, load_cifar10, load_cifar100c, load_cifar10, load_imagenetc
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel
from utils.imagenetloader import CustomImageFolder

import tent
import copy

from utils import get_imagenet_r_mask 


torch.manual_seed(0)

from tent import copy_model_and_optimizer, load_model_and_optimizer, softmax_entropy

torch.backends.cudnn.enabled=False


logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_cfg_fom_args('"ImageNet evaluation.')
logger.info("test-time adaptation:")

imagenet_r_mask = get_imagenet_r_mask()


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


def meta_test_adaptive(model, test_loader, n_inner_iter=1, adaptive=True, num_classes=1000):
    model = tent.configure_model(model)

    params, _ = tent.collect_params(model)
    inner_opt = setup_optimizer(params)

    if not adaptive:
        model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)

    acc = 0.
    counter = 0 
    num_examples = 0

    iterator = tqdm.tqdm(test_loader)
    for batch_data in iterator:
        if cfg.TEST.DATASET == "imagenetc":
            x_curr, y_curr, _ = batch_data 
        elif cfg.TEST.DATASET == "imagenetr":
            x_curr, y_curr = batch_data 

        counter += 1

        num_examples += x_curr.shape[0]
        if counter % 50 == 0:
            print("batch id ", counter)

        if not adaptive:
            load_model_and_optimizer(model, inner_opt,
                                 model_state, optimizer_state)

        x_curr, y_curr = x_curr.cuda(), y_curr.cuda()
        y_curr = y_curr.type(torch.cuda.LongTensor)

        for _ in range(n_inner_iter):
            T = cfg.OPTIM.TEMP
            eps = cfg.MODEL.EPS

            outputs = model(x_curr)

            if num_classes == 200:
                outputs = outputs[:, imagenet_r_mask]

            outputs = outputs / T
            if cfg.OPTIM.ADAPT == "ent":
                tta_loss = softmax_entropy(outputs)
            elif cfg.OPTIM.ADAPT == "rpl":
                p = F.softmax(outputs, dim=1)
                y_pl = outputs.max(1)[1]
                Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
                tta_loss = (1- (Yg**0.8))/0.8
            elif cfg.OPTIM.ADAPT == "conjugate":
                softmax_prob = F.softmax(outputs, dim=1)
                smax_inp = softmax_prob 

                eye = torch.eye(num_classes).to(outputs.device)
                eye = eye.reshape((1, num_classes, num_classes))
                eye = eye.repeat(outputs.shape[0], 1, 1)
                t2 = eps * torch.diag_embed(smax_inp)
                smax_inp = torch.unsqueeze(smax_inp, 2)
                t3 = eps*torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
                matrix = eye + t2 - t3
                y_star = torch.linalg.solve(matrix, smax_inp)
                y_star = torch.squeeze(y_star)

                pseudo_prob = y_star
                tta_loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob *(1-softmax_prob)).sum(dim=1)
            elif cfg.OPTIM.ADAPT == "softmax_pl":
                softmax_prob = F.softmax(outputs, dim=1)

                tta_loss = torch.logsumexp(outputs, dim=1) - (softmax_prob * outputs - eps * softmax_prob *(1-softmax_prob)).sum(dim=1)
            elif cfg.OPTIM.ADAPT == "hard_pl":
                yp = outputs.max(1)[1]
                eps=8
                y_star = 1 * F.one_hot(yp, num_classes=num_classes)
                thresh_idxs = torch.where(outputs.softmax(1).max(1)[0] > 0.75)
                tta_loss = torch.logsumexp(outputs[thresh_idxs], dim=1) - torch.sum(y_star[thresh_idxs]*outputs[thresh_idxs], dim=1) + torch.sum(eps*y_star[thresh_idxs]*(1 - F.softmax(outputs[thresh_idxs], dim=1)), dim=1)
            else:
                tta_loss = None

            tta_loss = tta_loss.mean()

            inner_opt.zero_grad()
            tta_loss.backward()

            inner_opt.step()

        outputs_new = model(x_curr)

        if num_classes == 200:
            outputs_new = outputs_new[:, imagenet_r_mask]

        acc += (outputs_new.max(1)[1] == y_curr).float().sum()

    return acc.item() / num_examples

def get_imagenetc_loader(data_dir, corruption, severity, batch_size, shuffle=False):
    data_folder_path = Path(data_dir) / "ImageNet-C"/ corruption / str(severity)

    prepr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    imagenet = CustomImageFolder(data_folder_path, prepr)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=20)
    return test_loader 

def get_imagenetr_loader(data_dir, batch_size, shuffle=False):
    prepr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    imagenet_r = datasets.ImageFolder(root=data_dir, transform=prepr)

    test_loader = data.DataLoader(imagenet_r,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=4,
                                  pin_memory=True)
    return test_loader 

err_array = np.zeros((len(cfg.CORRUPTION.SEVERITY)+1, len(cfg.CORRUPTION.TYPE)+1))
save_path = os.path.join(cfg.LOG_DIR, "adapt_%s_opt_%s_lr_%.1e_T_%.1f.txt"%(cfg.OPTIM.ADAPT, cfg.OPTIM.METHOD, cfg.OPTIM.LR, cfg.OPTIM.TEMP))
np.savetxt(save_path, err_array, fmt="%.4f")


for i, severity in enumerate(cfg.CORRUPTION.SEVERITY):
    err_list = []
    for j, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
        if cfg.TEST.DATASET == "imagenetc":
            test_loader = get_imagenetc_loader("/project_data/datasets", corruption_type, severity, cfg.TEST.BATCH_SIZE, False)
            num_classes = 1000
        elif cfg.TEST.DATASET == "imagenetr":
            test_loader = get_imagenetr_loader("/project_data/datasets/imagenet-r", cfg.TEST.BATCH_SIZE, False)
            num_classes = 200

        print("Meta test begin!")
        net_test = copy.deepcopy(net)

        acc = meta_test_adaptive(net_test, test_loader, 1, adaptive=True, num_classes=num_classes)

        print("Meta test finish!")
        err = 1. - acc
        err_list.append(err)

        logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")