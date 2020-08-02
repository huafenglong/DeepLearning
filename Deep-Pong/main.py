from __future__ import print_function, division
import os

import utils

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.multiprocessing as mp
from environment import atari_env
from py_argarse import args
from utils import read_config
from utils import TrafficLight
from utils import Counter,Shared_grad_buffers
from model import A3Clstm
from train import train
# from test import test
# from shared_optim import SharedRMSprop, SharedAdam
# #from gym.configuration import undo_logger_setup
import time
from chief import chief
#undo_logger_setup()


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    # --------设置global随机种子和多线程---------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    mp.set_start_method('spawn')

    # --------gym环境预处理---------------
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    envs = [atari_env(args.env, env_conf, args ,rank) for rank in range(args.workers)]
    # -------公用的lstm神经网络，load参数，放入公共内存---------------
    shared_model = A3Clstm(envs[0].observation_space.shape[0], envs[0].action_space)
    #放入公共内存:它可以不需要任何其他复制操作的发送到其他的进程中。
    shared_model.share_memory()
    # -------定义优化器 放入公共内存---------------
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    # -------定义控制信号---------------
    traffic_light = TrafficLight()
    counter = Counter()
    shared_grad_buffers = Shared_grad_buffers(shared_model)
    # -------定义进程 test和train---------------
    processes = []

    p = mp.Process(target=chief, args=(args.workers,traffic_light, counter, shared_model, shared_grad_buffers, optimizer))
    p.start()
    processes.append(p)

    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model,env_conf,traffic_light, counter,shared_grad_buffers))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()

