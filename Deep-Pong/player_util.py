from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    def __init__(self, env, args):
        self.model = None
        self.env = env
        self.args = args
        self.gpu_id = 0

        self.state = None
        self.hx = None
        self.cx = None

        self.states = []
        self.hxs = []
        self.cxs = []
        self.values = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = None
    '''
    通过状态，短期记忆，长期记忆输入到LSTM获取action的linear，critic的value,下一次的hx，cx
    取aciton.linear的softmax，logsoftmax获取动作的概率分布
    保存每个state action的entropy.sum
    entropy[n] = entropy[0] +.....+entropy[n-1]+entropy[n]
    保存具体动作的值 critic.value action.log_probs acion.rewards 
    保存entropy 第某个action时，动作概率的分布有多扩散
        entropy = 0 代表 1 0的分布
        entropy = max 代表 0.5 0.5的分布
    '''
    def action_train(self):
        #保存每一个step的输入状态
        self.states.append(self.state)
        self.hxs.append(self.hx.detach())
        self.cxs.append(self.cx.detach())
        #根据policy计算value和log，以及memory output
        value, logit, (self.hx, self.cx) = self.model((self.state, (self.hx, self.cx)))
        # 保存C model :values
        self.values.append(value)
        #根据logit,获得prob and action并保存
        prob = F.softmax(logit, dim=1).detach()
        action = prob.multinomial(1)  # multinomial按权重取最大值，次数为1
        log_prob = torch.log(prob.gather(1, action))
        self.actions.append(action)
        self.log_probs.append(log_prob)
        #next setp
        state, reward, done, _ = self.env.step(action.cpu().numpy())

        self.state = torch.tensor(state).float()
        reward = [max(min(r, 1), -1) for r in reward]
        reward = torch.tensor(reward).float()

        with torch.cuda.device(self.gpu_id):
            self.state = self.state.cuda()
            reward = reward.cuda()

        self.rewards.append(reward)
        self.dones.append(list(done))

        return self

    def clear_actions(self):
        self.states = []
        self.hxs = []
        self.cxs = []
        self.values = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []

        return self

