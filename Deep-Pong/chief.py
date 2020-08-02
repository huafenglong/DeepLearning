import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
import time

def chief(workers, traffic_light, counter, shared_model, shared_grad_buffers, optimizer):
    while True:
        # workers will wait after last loss computation
        if counter.get() == workers:
            #print(f"counter.get() is {counter.get()}")
            #print(shared_grad_buffers.grads['mu.weight_grad'])
            for n,p in shared_model.named_parameters():
                p._grad = Variable(shared_grad_buffers.grads[n+'_grad']/workers)
            optimizer.step()
            counter.reset()
            shared_grad_buffers.reset()
            traffic_light.switch() # workers start new loss computation
            #print('update')
