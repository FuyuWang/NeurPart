'''
Two action at a time discrete
'''
import math

import torch.nn as nn

from torch.distributions import Categorical
import numpy as np
import random
import copy

import torch
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Agent():
    def __init__(self, actor, h_size=128, device=None, is_gemm=False):
        self.is_gemm = is_gemm
        if is_gemm:
            state_div = [3, 1, 2, 1]
        else:
            state_div = [7,1, 2,1]
        self.state_div =np.cumsum(state_div)
        self.actor = actor
        self.device = device
        self.h_size = h_size
        self.lstm_hid_value = self.init_hidden_lstm()
        self.saved_log_probs = []
        self.rewards = []
        self.baseline = None
        self.lowest_reward = 0
        self.best_reward_whole_eps = float("-Inf")
        self.has_succeed_history = False
        self.bad_counts = 0
        self.learned_input = None

    def step(self, state, actions, log_prob, reward, next_state, done, sig, impt, infos):

        self.rewards.append(reward)
        self.saved_log_probs.append(log_prob)

    def save_hidden_lstm(self):
        self.lstm_hid = copy.deepcopy(self.actor.lstm.state_dict())
    def load_hidden_lstm(self):
        self.actor.lstm.load_state_dict(self.lstm_hid)
        del self.lstm_hid
    def init_hidden_lstm(self):
        return (torch.zeros(1, self.h_size, device=self.device),
                torch.zeros(1, self.h_size, device=self.device))

    def act(self, state, infos, eps=0.0, temperature=1, params=None, pe_constraints={}):

        dimensions = state[0:self.state_div[0]]
        action_status = state[self.state_div[0]:self.state_div[1]]
        actions = state[self.state_div[1]:self.state_div[2]]

        action_step = state[self.state_div[2]:self.state_div[3]]

        dimensions = torch.from_numpy(dimensions).type(torch.FloatTensor).to(device)
        action_status = torch.from_numpy(action_status).type(torch.FloatTensor).to(device)
        actions = torch.from_numpy(actions).type(torch.FloatTensor).to(device)
        action_step = torch.from_numpy(action_step).type(torch.LongTensor).to(device)

        p1, p2, self.lstm_hid_value = self.actor(dimensions, action_status, actions, action_step, self.lstm_hid_value,
                                                 temperature=temperature, params=params, pe_constraints=pe_constraints)

        res = torch.cat((self.lstm_hid_value[0], p1, p2), dim=1)
        if self.learned_input is None:
            self.learned_input = res
        else:
            self.learned_input = torch.cat((self.learned_input, res), dim=0)

        policy1 = Categorical(p1)
        action1 = policy1.sample()
        log_prob1 = policy1.log_prob(action1)

        policy2 = Categorical(p2)
        action2 = policy2.sample()
        log_prob2 = policy2.log_prob(action2)
        action = torch.stack([action1, action2], dim=0)
        log_prob = torch.stack([log_prob1, log_prob2], dim=0)

        return action.data, log_prob

    def reset(self):
        self.learned_input = None
        self.saved_log_probs = []
        self.rewards = []
        self.lstm_hid_value = self.init_hidden_lstm()
    
    def reset_lstm(self):
        self.lstm_hid_value = self.init_hidden_lstm()
    
    def set_fitness(self, fitness="energy"):
        self.fitness = fitness

    def minmax_reward(self, reward):
        self.lowest_reward = max(self.lowest_reward, np.min(reward))
        reward -= self.lowest_reward

        return reward


    def modify_reward(self, reward):
        pick_idx = reward != 0
        pick = reward[pick_idx]
        if len(pick)>0:
            reward[pick_idx] = (reward[pick_idx] - reward[pick_idx].mean())/(reward[pick_idx].std() + EPISIOLON)


    def impt_adj_reward(self, reward, impt):
        if impt is not None:
            reward[:len(impt)] = reward[:len(impt)] * impt
        return reward

    def backup(self, infos):
        if infos["succeed"]:
            self.has_succeed_history = True
            reward_whole_eps = infos["reward_whole_eps"]
            if self.best_reward_whole_eps < reward_whole_eps:
                self.best_reward_whole_eps = reward_whole_eps
                self.actor_backup = copy.deepcopy(self.actor.state_dict())
                self.bad_counts = 0
            else:
                self.bad_counts += 1
        elif self.has_succeed_history:
            self.bad_counts += 1
    def actor_refresh(self,  refresh_threshold=50):
        if self.bad_counts >refresh_threshold:
            self.actor.load_state_dict(self.actor_backup)
            self.bad_counts = 0
    
    def get_chkpt(self):
        chkpt = {"actor":self.actor.state_dict(),
                "best_actor": None,
                "baseline": self.baseline,
                "scheduler": self.scheduler,
                "optimizer": self.actor_optimizer.state_dict()}
        return chkpt
    def load_actor(self, chkpt):
        self.actor.load_state_dict(chkpt["actor"])

    def get_learned_input(self):
        return self.learned_input
    def reset_learned_input(self):
        self.learned_input = None

    def learned_loss(self, H, params=None):
        return self.actor.learned_loss(H, params)

