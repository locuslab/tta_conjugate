from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from pdb import set_trace as st


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, params=None, use_gram=False, g_train=None, classwise=False, meta_train=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic 
        self.params = params
        self.use_gram = use_gram
        self.g_train = g_train
        self.classwise = classwise
        self.meta_train = meta_train

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, y=None, loss_weight=None):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            if self.meta_train:
                loss_weight = forward_and_adapt_gram_meta_train(x, y, loss_weight, [0.001, 10, 100], self.model, self.optimizer, self.params, self.g_train)
                return loss_weight
            elif not self.use_gram:
                outputs = forward_and_adapt(x, self.model, self.optimizer)
            elif not self.classwise:
                outputs = forward_and_adapt_gram(x, self.model, self.optimizer, self.g_train)
            else:
                outputs = forward_and_adapt_gram_classwise(x, self.model, self.optimizer, self.g_train)

        self.model.eval()
        outputs = self.model(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor, T1: float=1, T2: float=1) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x1 = x / T1
    x2 = x / T2
    return -(x1.softmax(1) * x2.log_softmax(1)).sum(1)

@torch.enable_grad()
def compute_gram_matrix(input):
    a, b, c = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a, b * c)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    return G

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_gram_classwise(x, model, optimizer, g_train):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs, outputs_fp = model(x, feature_maps=True)
    class_preds = outputs.argmax(1)

    # adapt
    loss = 0
    loss_fn = nn.MSELoss()

    loss_weight = [0, 1, 0]

    loss_list = [0,0,0]
    for i in range(outputs.shape[0]):
        class_id = class_preds[i].item()
        loss += loss_weight[0] * loss_fn(compute_gram_matrix(outputs_fp[0][i]), g_train[0][class_id])
        loss += loss_weight[1] * loss_fn(compute_gram_matrix(outputs_fp[1][i]), g_train[1][class_id])
        loss += loss_weight[2] * loss_fn(compute_gram_matrix(outputs_fp[2][i]), g_train[2][class_id])

    loss /= outputs.shape[0]
    # for i in range(3):
    #     loss_list[i] += loss_fn(g_test[i], g_train[i])
    #     loss += loss_weight[i] * loss_list[i]

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return outputs

def loss_wishart(g_test, g_train):
    loss=0
    d_list = [1024,256,64]
    for idx in range(3):
        q = g_test[idx].shape[0]
        d = d_list[idx]
        t1 = torch.mm(g_test[idx],torch.inverse(g_train[idx]))
        t2 = torch.mm(g_train[idx],torch.inverse(g_test[idx]))
        t3 = torch.trace(t1+t2)
        t3 = t3 * q/4 - q*d/2
        if idx==1:
            t3=t3*4
        if idx==2:
            t3=t3*0
        loss+=t3
    
    return loss

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_gram(x, model, optimizer, g_train):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs, outputs_fp = model(x, feature_maps=True)

    m = nn.Softmax(dim=1)
    outputs_softmax = m(outputs)
    class_prob = outputs_softmax.max(1)[0]

    g_test = [torch.zeros(64, 64).to("cuda"),
            torch.zeros(128, 128).to("cuda"),
            torch.zeros(256, 256).to("cuda")]

    count = 0
    for i in range(outputs.shape[0]):
        if class_prob[i] >= 0.0:
            count += 1
            g_test[0] += compute_gram_matrix(outputs_fp[0][i])
            g_test[1] += compute_gram_matrix(outputs_fp[1][i])
            g_test[2] += compute_gram_matrix(outputs_fp[2][i])

    for i in range(3):
        # g_test[i] /= outputs.shape[0]
        g_test[i] /= count

    # adapt
    loss = 0
    loss_fn = nn.MSELoss()

    loss_weight = [-2, 10, 1]

    loss_list = [0,0,0]
    for i in range(3):
        loss_list[i] += loss_fn(g_test[i], g_train[i])
        loss += loss_weight[i] * loss_list[i]

    # loss = loss_wishart(g_test, g_train)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return outputs

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_gram_meta_train(x, y, loss_weight, lr, model, optimizer, params, g_train):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs, outputs_fp = model(x, feature_maps=True)

    m = nn.Softmax(dim=1)
    outputs_softmax = m(outputs)
    class_prob = outputs_softmax.max(1)[0]

    g_test = [torch.zeros(64, 64).to("cuda"),
            torch.zeros(128, 128).to("cuda"),
            torch.zeros(256, 256).to("cuda")]

    count = 0
    for i in range(outputs.shape[0]):
        if class_prob[i] >= 0.0:
            count += 1
            g_test[0] += compute_gram_matrix(outputs_fp[0][i])
            g_test[1] += compute_gram_matrix(outputs_fp[1][i])
            g_test[2] += compute_gram_matrix(outputs_fp[2][i])

    for i in range(3):
        # g_test[i] /= outputs.shape[0]
        g_test[i] /= count

    # adapt
    loss = 0
    loss_fn = nn.MSELoss()

    loss_list = [0,0,0]
    for i in range(3):
        loss_list[i] += loss_fn(g_test[i], g_train[i])
        loss += loss_weight[i] * loss_list[i]

    grads = [0, 0, 0]
    grads[0] = torch.autograd.grad(loss_list[0], params, retain_graph=True, allow_unused=True)
    grads[1] = torch.autograd.grad(loss_list[1], params, retain_graph=True, allow_unused=True)
    grads[2] = torch.autograd.grad(loss_list[2], params, retain_graph=True, allow_unused=True)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    CE_Loss = nn.CrossEntropyLoss()
    outputs_new = model(x)
    
    meta_loss = CE_Loss(outputs_new, y)

    grads_new = torch.autograd.grad(meta_loss, params, allow_unused=True)
    
    values = [0, 0, 0]
    for i in range(3):
        for (a, b) in zip(grads[i], grads_new):
            if a is not None:
                # a = (a - torch.mean(a))/torch.var(a)
                # st()
                values[i] += torch.dot(a, b)

    for i in range(3):
        loss_weight[i] += lr[i] * values[i]
    return loss_weight

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def collect_params_full(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if not isinstance(m, nn.BatchNorm2d):
        if True:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes

            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def configure_model_eval(model):
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

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"