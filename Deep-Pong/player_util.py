from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = 0
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
        #单步的action
        value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        #计算每次的概率和entropy(entropies)和entropy的sum,sum是每一步所有动作概率的熵值
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        #通过prob 采样对应的动作和动作logprob
        action = prob.multinomial(1).data   #multinomial按权重取最大值，次数为1
        log_prob = log_prob.gather(1, Variable(action))
        #next setp
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        with torch.cuda.device(self.gpu_id):
            self.state = self.state.cuda()
        #reward (-1 to 1),gym env优化处理的结果
        self.reward = max(min(self.reward, 1), -1)
        #保存list：critic.value action.log_probs acion.rewards
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    #test任务时，做测试不训练
    def action_test(self):
        #可能时LSTM的需求，cx和hx
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1)[1].data.cpu().numpy()
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
