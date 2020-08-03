from __future__ import print_function, division
import os

import utils

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.multiprocessing as mp
from environment import atari_env, ParallelEnv
from py_argarse import args
from utils import read_config

from train import train
from player_util import Agent
# from test import test
# from shared_optim import SharedRMSprop, SharedAdam
# #from gym.configuration import undo_logger_setup
import time
#undo_logger_setup()
from torch.autograd import Variable


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior
if __name__ == '__main__':
    # --------设置global随机种子和多线程---------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # --------gym环境预处理---------------
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    envs = [atari_env(args.env, env_conf, args, rank) for rank in range(args.workers)]
    observation_space, action_space = envs[0].observation_space.shape[0], envs[0].action_space
    # -------公用的lstm神经网络，load参数
    envs = ParallelEnv(envs)

    train(args, envs, observation_space, action_space)








