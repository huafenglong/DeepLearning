from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable


def train(rank, args, shared_model, env_conf,traffic_light, counter,shared_grad_buffers):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = 0
    env = atari_env(args.env, env_conf, args)
    # 每个worker的随机种子
    torch.manual_seed(args.seed + rank)
    env.seed(args.seed + rank)

    #每个单独的work,独立的环境和model，在cuda中运行
    player = Agent(None, env, args, None)
    player.model = A3Clstm(env.observation_space.shape[0], env.action_space)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()

    with torch.cuda.device(gpu_id):
        player.state = player.state.cuda()
        player.model = player.model.cuda()

    #切换到训练模式
    player.model.train()

    while True:
        signal_init = traffic_light.get()

        #读取公共model的参数，shared_model在share_memory中可以直接使用（不会修改它）
        with torch.cuda.device(gpu_id):
            player.model.load_state_dict(shared_model.state_dict())

        #game over时记忆值重置
        if player.done:
            with torch.cuda.device(gpu_id):
                player.cx = Variable(torch.zeros(1, 512).cuda())
                player.hx = Variable(torch.zeros(1, 512).cuda())
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        #训练20步或者game over就结束训练
        for step in range(args.num_steps):
            #训练时，保存每一步的相关信息到list
            player.env.render()
            player.action_train()
            if player.done:
                break

        #训练结束，重置state到cuda
        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            with torch.cuda.device(gpu_id):
                player.state = player.state.cuda()

        #第n+1步的state value:next_state   计算advantages
        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            R = value.data

        with torch.cuda.device(gpu_id):
            R = R.cuda()

        player.values.append(Variable(R))

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()

        #第 n+1 步的value
        R = Variable(R)
        #i form n to  1
        for i in reversed(range(len(player.rewards))):
            #advantage[0:n]
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

            # Generalized Advantage Estimataion
            # 第n次的td_error 到第0次的td——error
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            # 第n次的td_error 到第0次的td——error TD(&)更新公式
            gae = gae * args.gamma * args.tau + delta_t
            #policy_loss取sum，log_probs*TD(&)advantage，entropies
            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        player.model.zero_grad()
        # 反向传播计算player的grads
        (policy_loss + 0.5 * value_loss).backward()
        # 将workers的grads推送给global网络
        #ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        #player.optimizer.step()    #更新global网络的grads
        shared_grad_buffers.add_gradient(player.model)
        player.clear_actions()

        counter.increment()
        # wait for a new signal to continue
        while traffic_light.get() == signal_init:
            pass
