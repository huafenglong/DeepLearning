from __future__ import division
import numpy as np
import json
import logging
import torch
import torch.multiprocessing as mp

class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)

class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0

class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data.cpu()

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
