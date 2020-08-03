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

def train(args, envs, observation_space, action_space):
    gpu_id = 0
    #每个单独的work,独立的环境和model，在cuda中运行
    player = Agent(None, envs, args, None)

    player.model = A3Clstm(observation_space, action_space)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()

    with torch.cuda.device(gpu_id):
        player.state = player.state.cuda()
        player.model = player.model.cuda()

    optimizer = torch.optim.Adam(player.model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    #切换到训练模式
    player.model.train()

    while True:
        #game over时记忆值重置
        player.cx = Variable(player.cx.data)
        player.hx = Variable(player.hx.data)

        for index in range(args.workers):
            if player.done[index]:
                with torch.cuda.device(gpu_id):
                    player.cx[index] = Variable(torch.zeros(512).cuda())
                    player.hx[index] = Variable(torch.zeros(512).cuda())

        #训练20步或者game over就结束训练
        for step in range(args.num_steps):
            #训练时，保存每一步的相关信息到list
            player.env.get_images()
            player.action_train()
            if player.done.any():
                break

        #第n+1步的state value:next_state   计算advantages
        R = torch.zeros(args.workers, 1)
        value, _, _ = player.model((Variable(player.state), (player.hx, player.cx)))
        for index in range(args.workers):
            if not player.done[index]:
                R[index] = value[index].data

        with torch.cuda.device(gpu_id):
            R = R.cuda()

        player.values.append(Variable(R))

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(args.workers, 1)
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()

        #第 n+1 步的value
        R = Variable(R)
        #i form n to  1
        for i in reversed(range(len(player.rewards))):
            # advantage[0:n]
            # 第0，1，2，...n 到 n+1的估值差 r[0-n],r[1-n],r[2-n]....rn   Value(N+1) 取反：
            # 第n,n-1,n-2,n-3,......3,2,1
            # r[n] + Value(N+1) - Value(N)
            # r[n:n-1] + Value(N+1) - Value(N-1)
            # ...
            # r[n:2] + Value(N + 1) - Value(2)
            # r[n:1] + Value(N + 1) - Value(1)
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            #value_loss将所有的loss取平方并乘以0.5即：MessLoss
            value_loss = value_loss + 0.5 * advantage.pow(2)

            #Generalized Advantage Estimataion
            #第n次的td_error 到第0次的td——error
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            #第n次的td_error 到第0次的td——error TD(&)更新公式
            gae = gae * args.gamma * args.tau + delta_t

            # ratio = torch.exp(action_log_probs -
            #                   old_action_log_probs_batch)
            # surr1 = ratio * adv_targ
            # surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
            #                     1.0 + self.clip_param) * adv_targ
            # action_loss = -torch.min(surr1, surr2).mean()

            #policy_loss取sum，log_probs*TD(&)advantage，entropies
            policy_loss = policy_loss - player.log_probs[i] * Variable(gae) - 0.01 * player.entropies[i]

        player.model.zero_grad()
        #optimizer.zero_grad()
        # 反向传播计算player的grads
        (policy_loss + 0.5 * value_loss).mean().backward()
        optimizer.step()
        player.clear_actions()


