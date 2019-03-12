import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        """A simple DQN agent"""
        nn.Module.__init__(self)
        self.epsilon = epsilon
        self.n_actions = n_actions
        img_c, img_w, img_h = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        self.features = torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.classify = torch.nn.Sequential(
                torch.nn.Linear(4096, 1024),
                torch.nn.ReLU(True),
                torch.nn.Linear(1024, self.n_actions)
        )
        self.softmax = torch.nn.Softmax(1)
    
    def forward(self, state_t):
        shape = state_t.shape
        #plot_state(x)
        #print(x.shape)
        #x = torch.from_numpy(state_t.reshape(1, 1, *shape)).float()
        x = self.features(state_t)
        #print(x.shape)
        x = x.view(-1, 4096)
        x = self.classify(x)
        qvalues = self.softmax(x)
        #print(qvalues)
        assert isinstance(qvalues, Variable) and qvalues.requires_grad, "qvalues must be a torch variable with grad"
        assert len(qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions
        
        return qvalues #torch.argmax(qvalues).numpy()


    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not Variables
        """
        states = Variable(torch.FloatTensor(np.asarray(states)))
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
