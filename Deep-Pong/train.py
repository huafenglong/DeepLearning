from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import gym
import torch.nn.functional as F

def train(args, envs, observation_space, action_space):
    gpu_id = 0
    #每个单独的work,独立的环境和model，在cuda中运行
    player = Agent(envs, args)

    player.model = A3Clstm(observation_space, action_space)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()

    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()
        player.state = player.state.cuda()
        player.cx = torch.zeros(args.workers, 512).cuda()
        player.hx = torch.zeros(args.workers, 512).cuda()

    optimizer = torch.optim.Adam(player.model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    #切换到训练模式
    player.model.train()
    while True:
        #训练20步或者game over就结束训练
        for step in range(args.num_steps):
            #训练时，保存每一步的相关信息到list
            player.env.get_images()
            player.action_train()
            if player.dones[-1][0]:
                break

        if not player.dones[-1][0]:
            value, _, _ = player.model((player.state,(player.hx, player.cx)))
            R = value.detach()
        else:
            R = torch.zeros(args.workers, 1)
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(R)

        for j in range(args.num_ppo_train):
            policy_loss = 0
            value_loss = 0
            gae = 0

            for i in reversed(range(len(player.rewards))):
                value, logit, _ = player.model((player.states[i], (player.hxs[i], player.cxs[i])))
                prob = F.softmax(logit, dim=1)
                log_prob = F.log_softmax(logit, dim=1)
                entropy = -(log_prob * prob).sum(1)
                log_probs_current = log_prob.gather(1, player.actions[i])

                R = args.gamma * R + player.rewards[i]

                advantage = R - value
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = player.rewards[i] + args.gamma * player.values[i + 1].detach() - player.values[i].detach()
                gae = gae * args.gamma * args.tau + delta_t

                ratio = torch.exp(log_probs_current - player.log_probs[i])
                surr1 = ratio
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)

                policy_loss = policy_loss - torch.min(surr1, surr2) * gae * - 0.01 * entropy

            optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).mean().backward()
            optimizer.step()

        #game over时记忆值重置
        if player.dones[-1][0]:
            with torch.cuda.device(gpu_id):
                player.cx = torch.zeros(args.workers, 512).cuda()
                player.hx = torch.zeros(args.workers, 512).cuda()
        else:
            player.cx = player.cx.detach()
            player.hx = player.hx.detach()

        player.clear_actions()

# advantage[0:n]
# 第0，1，2，...n 到 n+1的估值差 r[0-n],r[1-n],r[2-n]....rn   Value(N+1) 取反：
# 第n,n-1,n-2,n-3,......3,2,1
# r[n] + Value(N+1) - Value(N)
# r[n:n-1] + Value(N+1) - Value(N-1)
# ...
# r[n:2] + Value(N + 1) - Value(2)
# r[n:1] + Value(N + 1) - Value(1)
# R = args.gamma * R + player.rewards[i]
# advantage = R - player.values[i]
# value_loss = value_loss + 0.5 * advantage.pow(2)
# value_loss = 0.5 * advantage.pow(2)
# advantage = args.gamma * R + player.rewards[i] - player.values[i]

        #entropy = -(log_prob * prob).sum(1)
        #self.entropies.append(entropy)
        #通过prob 采样对应的动作和动作logprob
# 计算每次的概率和entropy(entropies)和entropy的sum,sum是每一步所有动作概率的熵值