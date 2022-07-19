import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import logging 

import os 
import numpy as np

from models import * 
from conf import cfg, load_cfg_fom_args

from robustbench.data import load_cifar10c, load_cifar100c

import tent
import copy

import time

torch.manual_seed(0)

from tent import copy_model_and_optimizer, load_model_and_optimizer, softmax_entropy

torch.backends.cudnn.enabled=False

from pdb import set_trace as st 

logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


load_cfg_fom_args('"CIFAR-10-C evaluation.')
logger.info("test-time adaptation: TENT")

if not os.path.exists(cfg.LOG_DIR):
    os.makedirs(cfg.LOG_DIR)

if cfg.CORRUPTION.DATASET == "cifar10":
    ckpt_path = cfg.MODEL.CKPT_PATH

    net = Normalized_ResNet(depth=26)
    checkpoint = torch.load(ckpt_path)
    checkpoint = checkpoint['net']

    net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    net.load_state_dict(checkpoint)

elif cfg.CORRUPTION.DATASET == "cifar100":
    ckpt_path = cfg.MODEL.CKPT_PATH
    net = Normalized_ResNet_CIFAR100()
    net = torch.nn.DataParallel(net)

    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint["net"])

    net.to(device)
    cudnn.benchmark = True

def test_clean(model, x_test, y_test, batch_size):
    acc = 0.
    model.eval()

    n_batches = math.ceil(x_test.shape[0] / batch_size)
    for counter in range(n_batches):
        x_curr = x_test[counter * batch_size:(counter + 1) *
                   batch_size].to(device)
        y_curr = y_test[counter * batch_size:(counter + 1) *
                   batch_size].to(device)

        outputs = model(x_curr)
        acc += (outputs.max(1)[1] == y_curr).float().sum()

    return acc.item() / x_test.shape[0]

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.eval()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
    return model

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

def meta_test_adaptive(model, x_test, y_test, batch_size,  n_inner_iter=1, adaptive=True, use_test_bn=True, num_classes=10):
    if use_test_bn:
        model = tent.configure_model(model)
    else:
        model = tent.configure_model_eval(model)

    params, _ = tent.collect_params(model)
    inner_opt = setup_optimizer(params)

    if not adaptive:
        model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)

    acc = 0.
    n_batches = math.ceil(x_test.shape[0] / batch_size)

    for counter in range(n_batches):
        if not adaptive:
            load_model_and_optimizer(model, inner_opt,
                                 model_state, optimizer_state)

        x_curr = x_test[counter * batch_size:(counter + 1) * batch_size].to(device)
        y_curr = y_test[counter * batch_size:(counter + 1) * batch_size].to(device)

        for _ in range(n_inner_iter):
            outputs = model(x_curr)
            outputs = outputs / cfg.OPTIM.TEMP
            softmax_prob = F.softmax(outputs, dim=1)
            eps = cfg.MODEL.EPS

            if cfg.OPTIM.ADAPT == "ent":
                tta_loss = softmax_entropy(outputs)

            elif cfg.OPTIM.ADAPT == "conjugate":
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
                tta_loss = torch.logsumexp(outputs, dim=1) - (softmax_prob * outputs - eps * softmax_prob * (1-softmax_prob)).sum(dim=1)
            elif cfg.OPTIM.ADAPT == "hard_pl":
                yp = outputs.max(1)[1]
                y_star = 1 * F.one_hot(yp, num_classes=num_classes)
                thresh_idxs = torch.where(outputs.softmax(1).max(1)[0] > 0.75)
                tta_loss = torch.logsumexp(outputs[thresh_idxs], dim=1) - torch.sum(y_star[thresh_idxs]*outputs[thresh_idxs], dim=1) + torch.sum(eps*y_star[thresh_idxs]*(1 - F.softmax(outputs[thresh_idxs], dim=1)), dim=1)
            elif cfg.OPTIM.ADAPT == "rpl":
                p = F.softmax(outputs, dim=1)
                y_pl = outputs.max(1)[1]
                Yg = torch.gather(p, 1, torch.unsqueeze(y_pl, 1))
                tta_loss = (1- (Yg**0.8))/0.8
            else:
                pass 

            tta_loss = tta_loss.mean()

            inner_opt.zero_grad()
            tta_loss.backward()

            inner_opt.step()

        outputs_new = model(x_curr)
        acc += (outputs_new.max(1)[1] == y_curr).float().sum()

    return acc.item() / x_test.shape[0]

for i, severity in enumerate(cfg.CORRUPTION.SEVERITY):
    err_list = []
    for j, corruption_type in enumerate(cfg.CORRUPTION.TYPE):

        if cfg.CORRUPTION.DATASET == "cifar10":
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,severity, cfg.DATA_DIR, True, [corruption_type])
            num_classes=10
        elif cfg.CORRUPTION.DATASET == "cifar100":
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,severity, cfg.DATA_DIR, False, [corruption_type])
            num_classes=100
        else:
            print("ERROR: no valid datatset provided, must be cifar10 and cifar100")

        x_test, y_test = x_test.cuda(), y_test.cuda()
        y_test = y_test.type(torch.cuda.LongTensor)

        print("Meta test begin!")
        net_test = copy.deepcopy(net)

        acc = meta_test_adaptive(net_test, x_test, y_test, cfg.TEST.BATCH_SIZE, 1, adaptive=True, use_test_bn=True, num_classes=num_classes)
        err = 1. - acc
        logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
