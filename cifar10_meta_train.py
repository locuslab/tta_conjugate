import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import logging 
import higher

from models import *
from conf import cfg, load_cfg_fom_args
import tent
import copy

from tent import copy_model_and_optimizer, load_model_and_optimizer, softmax_entropy

from robustbench.data import load_cifar10c
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

import os 

torch.manual_seed(0)
torch.backends.cudnn.enabled=False


logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_cfg_fom_args('"CIFAR-10-C evaluation.')
logger.info("test-time adaptation: TENT")

if cfg.MODEL.ARCH == "ResNet-18":
    ckpt_path = cfg.MODEL.CKPT_PATH

    net = Normalized_ResNet(depth=26)
    checkpoint = torch.load(ckpt_path)
    checkpoint = checkpoint['net']

    net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.load_state_dict(checkpoint)
else:
    net = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()


class meta_loss_transformer(nn.Module):

    def __init__(self, meta_in=10, transformer_input_dim=16, n_heads=2, dim_feedforward=64, activation="relu", num_probe_layers=1, softmax=False):
        super(meta_loss_transformer, self).__init__()

        probe_activation = torch.nn.ReLU 
        self.transformer_input_dim = transformer_input_dim
        max_seq_length = meta_in
        self.pos_emb = torch.nn.Parameter(torch.randn(max_seq_length, self.transformer_input_dim-1))

        self.pos_emb.requires_grad = True
        self.enc_layer = torch.nn.TransformerEncoderLayer(self.transformer_input_dim, n_heads, dim_feedforward=dim_feedforward,
                         dropout=0.1, activation=activation, batch_first=True)

        if num_probe_layers==1:
            self.loss_fn = torch.nn.Sequential(
                                nn.Linear(self.transformer_input_dim*max_seq_length, 1)
                            )

        if num_probe_layers==2:
            print("2 layers")
            self.loss_fn = torch.nn.Sequential(
                                nn.Linear(self.transformer_input_dim*max_seq_length, int(self.transformer_input_dim*max_seq_length/4)), probe_activation(),
                                nn.Linear(int(self.transformer_input_dim*max_seq_length/4), 1)
                            )

        self.softmax = softmax

    def forward(self, x): # x: B,10
        if self.softmax:
            x = F.softmax(x, dim=1)

        token_embed = torch.unsqueeze(x,2)
        pos_embed = self.pos_emb[:token_embed.shape[1],:]
        pos_embed = torch.unsqueeze(pos_embed, 0)
        pos_embed = pos_embed.repeat(token_embed.shape[0],1,1)

        transformer_inp = torch.cat((token_embed, pos_embed), dim=2)

        temp = self.enc_layer(transformer_inp)
        temp = temp.view(-1, temp.shape[1]*temp.shape[2])

        temp = self.loss_fn(temp)
        return temp

def meta_test(model, learnable_loss, x_test, y_test, batch_size, n_inner_iter=1, adaptive=True):
    model = tent.configure_model(model)

    params, _ = tent.collect_params(model)
    inner_opt = setup_optimizer(params)

    opt_2 = torch.optim.SGD(learnable_loss.parameters(), lr=0)

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
            loss_input = outputs 

            meta_loss = learnable_loss(loss_input)

            meta_loss = meta_loss.mean()

            opt_2.zero_grad()
            inner_opt.zero_grad()
            meta_loss.backward()

            inner_opt.step()
            opt_2.step()
            opt_2.zero_grad()

        outputs_new = model(x_curr)
        acc += (outputs_new.max(1)[1] == y_curr).float().sum()

    return acc.item() / x_test.shape[0]

def sample_examples(x_test, y_test, batch_size):
    # perm = torch.randperm(batch_size)
    perm = torch.randperm(x_test.shape[0])[:batch_size]
    x_sample = x_test[perm]
    y_sample = y_test[perm]

    return x_sample, y_sample

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

def meta_train_one_epoch(model, meta_opt, inner_opt, learnable_loss, x_test, y_test, batch_size, n_inner_iter=1):
    for param in learnable_loss.parameters():
        param.requires_grad = True

    n_batches = 50

    params, _ = tent.collect_params(model)
    l_opt = setup_optimizer(params, lr_test=None)
    l_loss_test = copy.deepcopy(learnable_loss)

    for counter in range(n_batches):
        x_train, y_train = sample_examples(x_test, y_test, batch_size)

        x_val, y_val = sample_examples(x_test, y_test, batch_size)

        outputs = model(x_train)
        loss_input = outputs 

        l_loss = l_loss_test(loss_input).mean()

        l_loss.backward(retain_graph=True)
        l_opt.step()
        l_opt.zero_grad() 

        if (counter+1) % 5 == 0:
            inner_opt.load_state_dict(l_opt.state_dict())

            with higher.innerloop_ctx(model, inner_opt) as (fmodel, diffopt):
                for _ in range(n_inner_iter):
                    outputs = fmodel(x_train)
                    loss_input = outputs 

                    meta_loss = learnable_loss(loss_input).mean()

                    diffopt.step(meta_loss)

                yp = fmodel(x_val)
                task_loss = F.cross_entropy(yp, y_val)
                task_loss.backward()

            meta_opt.step()
            meta_opt.zero_grad()

    return learnable_loss


learnable_loss = meta_loss_transformer(meta_in=10, transformer_input_dim=cfg.TRANSFORMER.INPUT_DIM, 
                n_heads=cfg.TRANSFORMER.N_HEADS, dim_feedforward=cfg.TRANSFORMER.DIM_FF, 
                activation=cfg.TRANSFORMER.ACTIVATION, num_probe_layers=cfg.TRANSFORMER.PROBE_LAYERS).cuda()

NUM_EPOCHS=50

err_list = np.zeros((NUM_EPOCHS+1, 1))

for severity in cfg.CORRUPTION.SEVERITY:
    for corruption_type in cfg.CORRUPTION.TYPE:
        x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,severity, cfg.DATA_DIR, False,[corruption_type])

        x_test, y_test = x_test.cuda(), y_test.cuda()
        y_test = y_test.type(torch.cuda.LongTensor)

        net_train = copy.deepcopy(net)
        model = tent.configure_model(net_train)
        params, param_names = tent.collect_params(model)

        inner_opt = setup_optimizer(params, lr_test=None)

        model_state, optimizer_state = copy_model_and_optimizer(model, inner_opt)

        meta_opt = torch.optim.Adam(learnable_loss.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=NUM_EPOCHS, verbose=True, eta_min=1e-6)

        num_examples_train, num_examples_test, num_inner_iter = 10000, 10000, 1

        net_test = copy.deepcopy(net)
        acc = meta_test(net_test, learnable_loss, x_test[:num_examples_test], y_test[:num_examples_test], cfg.TEST.BATCH_SIZE, num_inner_iter, True)
        err = 1.-acc 
        logger.info(f"before meta-train error: {err:.2%}")
        err_list[0][0] = err 

        best_err = 1.
        for epoch in range(NUM_EPOCHS):
            learnable_loss.train()
            load_model_and_optimizer(model, inner_opt, model_state, optimizer_state)

            learnable_loss = meta_train_one_epoch(model, meta_opt, inner_opt, learnable_loss, x_test[:num_examples_train], y_test[:num_examples_train], cfg.TEST.BATCH_SIZE, num_inner_iter)

            scheduler.step()

            if epoch % 1 == 0:
                learnable_loss.eval()
                net_test = copy.deepcopy(net)
                acc = meta_test(net_test, learnable_loss, x_test[:num_examples_test], y_test[:num_examples_test], cfg.TEST.BATCH_SIZE, num_inner_iter, True)
                err = 1.-acc
                err_list[epoch+1][0] = err

                save_path = "eval_results/meta_loss/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.savetxt(os.path.join(save_path, "log.txt"), err_list, fmt="%.4f")

                print(f"Meta Epoch {epoch} err: {err:.2%}")

                torch.save({"state_dict": learnable_loss.state_dict(), "error": err}, os.path.join(save_path, "epoch_%d.pth"%epoch))