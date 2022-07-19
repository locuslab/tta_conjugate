import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms

import logging 

import os 

from models import *
from conf import cfg, load_cfg_fom_args

from robustbench.model_zoo.enums import BenchmarkDataset

import tent
import copy

torch.manual_seed(0)

from tent import copy_model_and_optimizer, load_model_and_optimizer
from utils import load_corruptions_cifar, AugMixDataset

torch.backends.cudnn.enabled=False


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
    net = Normalized_ResNet_CIFAR100(depth=26)
    net = torch.nn.DataParallel(net)

    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint["net"])

    net.to(device)
    cudnn.benchmark = True
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


def meta_test_adaptive(model, test_loader, batch_size, adaptive=True, use_test_bn=True):
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

    for _, (images, y_curr) in enumerate(test_loader):
        if not adaptive:
            load_model_and_optimizer(model, inner_opt,
                                 model_state, optimizer_state)

        y_curr = y_curr.to(device)

        for _ in range(1):
            logits_aug1 = model(images[1].to(device))
            logits_aug2 = model(images[2].to(device))
            logits_aug3 = model(images[3].to(device))

            T = cfg.OPTIM.TEMP
            p_aug1, p_aug2, p_aug3 = F.softmax(logits_aug1/T, dim=1), F.softmax(logits_aug2/T, dim=1), F.softmax(logits_aug3/T, dim=1)

            p_avg = (p_aug1 + p_aug2 + p_aug3) / 3
            tta_loss = - (p_avg * torch.log(p_avg)).sum(dim=1)

            tta_loss = tta_loss.mean()

            inner_opt.zero_grad()
            tta_loss.backward()

            inner_opt.step()

        outputs_new = model(images[0])
        acc += (outputs_new.max(1)[1] == y_curr).float().sum()

    return acc.item() / x_test.shape[0]


for i, severity in enumerate(cfg.CORRUPTION.SEVERITY):
    err_list = []
    for j, corruption_type in enumerate(cfg.CORRUPTION.TYPE):

        if cfg.CORRUPTION.DATASET == "cifar10":
            x_test, y_test = load_corruptions_cifar(BenchmarkDataset.cifar_10, cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, [corruption_type], True)

            preprocess = transforms.Compose(
                [transforms.ToTensor()])
            test_data = AugMixDataset(x_test, y_test, preprocess, False)

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True)

        elif cfg.CORRUPTION.DATASET == "cifar100":
            x_test, y_test = load_corruptions_cifar(BenchmarkDataset.cifar_100, cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, [corruption_type], True)

            preprocess = transforms.Compose(
                [transforms.ToTensor()])
            test_data = AugMixDataset(x_test, y_test, preprocess, False)

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True)
        else:
            print("ERROR: no valid datatset provided, must be cifar10 and cifar100")

        print("Meta test begin!")
        net_test = copy.deepcopy(net)

        acc = meta_test_adaptive(net_test, test_loader, cfg.TEST.BATCH_SIZE, adaptive=True, use_test_bn=True)

        print("Meta test finish!")
        err = 1. - acc
        err_list.append(err)
        logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")

    mean_err = np.mean(err_list)
    logger.info(f"mean error is % {mean_err:.2%}")