
from subprocess import Popen, PIPE
import pandas as pd
import os
import random
import pickle
import copy
import os, sys
import math
import numpy as np

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

df_dict = {1:"dla", 2:"shi", 3:"eye"}
m_type_dicts = {1:"CONV", 2:"DSCONV", 3:"CONV", 4:"TRCONV"}


class MaestroEnvironment(object):
    def __init__(self, model_name, model_defs, finish_reward=100, dim_size=6,n_action_steps=2, resource_size=2,
                 fitness='latency', dataflow="dla",
                 num_pe=4096, l1_size=8000, l2_size=8000):
        super(MaestroEnvironment,self).__init__()
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        random_file_name = random.randint(0, 2 ** 31)
        self.random_file_name = "{}".format(random_file_name)

        self.fitness = fitness
        self.model_name = model_name
        self.state = np.array([0.5]*8)
        self.last_runtime = 2 ** 64
        self.last_energy = 2**64
        self.last_throughput = 1
        self.observation = [0,0, 0,0,0,0]
        self.resource_state = [0, 0]
        self.consecutive_fail = 0
        self.max_fail = 0

        self.sig = 1
        self.mac_rec = []
        self.epoch = 0
        self.sol = []
        self.sol_record = []
        self.dataflow = dataflow
        self.constraint = "buffer_size"
        self.constraint_value = np.array([num_pe, l1_size, l2_size])
        self.total_used_constraints = np.array([0., 0., 0.])
        self.exp_table = {}
        self.is_gemm = False

        self.best_sol = None
        self.n_action_steps = n_action_steps
        self.emptyinfo = [-1] * (len(self.observation) + len(self.resource_state))
        self.resource_size = resource_size
        self.dim_size = dim_size

        self.model2model_defs = {}
        model_defs = self.get_model_defs(model_name, model_defs)
        self.total_step = len(model_defs)
        self.model_defs = model_defs
        self.model_defs_saved = copy.deepcopy(model_defs)
        model_bound = np.max(model_defs, axis=0, keepdims=True)
        self.model_defs_norm = model_defs/model_bound
        self.model_defs_norm_saved = copy.deepcopy(self.model_defs_norm)

        self.model2l1_size_array = {}
        self.l1_size_array = self.get_l1_size_array(self.model_name, self.model_defs)
        self.l1_size_array_saved = copy.deepcopy(self.l1_size_array)

        self.model2min_eps_reward = {
            'googlenet': 0, 'mnasnet': 0, 'resnet18': 0, 'shufflenet_v2': 0, 'vgg16': 0,
            'resnet50': 0, 'unet': 0, 'transformer': 0, 'mobilenet_v2': 0, 'densenet': 0
        }
        self.whole_eps_rewards = []
        self.min_eps_reward = 0
        self.best_reward = float('-inf')
        self.batch_size = 512

        self.model2draw = {}
        self.model2baseline = {}

        self.draw = self.get_draw(self.model_name, self.model_defs)
        self.model_defs = self.model_defs_saved[self.draw]
        self.model_defs_norm = self.model_defs_norm_saved[self.draw]
        self.l1_size_array = self.l1_size_array_saved[self.draw]
        self.action_size, self.action_space, self.action_bound, self.action_bottom = self.get_action_space(num_pe,
                                                                                                           l1_size,
                                                                                                           model_name,
                                                                                                           model_defs)
        self.total_resources = self.action_bound
        self.action_bound = self.action_bound[:n_action_steps]

    def get_model_defs(self, model_name, model_defs):
        if model_name in self.model2model_defs:
            return self.model2model_defs[model_name]
        else:
            model_defs_saved = copy.deepcopy(model_defs)
            total_latency = []
            avg_pe = math.floor(self.constraint_value[0] / len(model_defs))
            for i in range(len(model_defs)):
                maestro_state = np.concatenate((model_defs_saved[i], np.array([avg_pe, 1])))
                reward, mac, constraint = self.oberserve_maestro(maestro_state)
                if not reward or reward is None:
                    return None
                else:
                    total_latency.append(-1 * reward[0])

            total_latency = np.array(total_latency)
            if total_latency.max() / total_latency.min() > 100:
                if total_latency.argmax() == 0:
                    model_defs = model_defs_saved[1:]
                elif total_latency.argmax() == len(model_defs_saved) - 1:
                    model_defs = model_defs_saved[0:-1]
                else:
                    model_defs = np.concatenate((model_defs_saved[0:total_latency.argmax()],
                                                 model_defs_saved[total_latency.argmax() + 1:]), axis=0)

            self.model2model_defs[model_name] = model_defs
            return model_defs

    def get_action_space(self, num_pe, l1_size, model_name, model_defs, use_default=True):
        action_size = [10, 10]
        num_layers, dim_size = model_defs.shape
        med_pe = 1. * num_pe / num_layers
        med_ktile_size = (l1_size - self.l1_size_array[:, 1].sum()) / (self.l1_size_array[:, 0].sum())
        pe_array = np.array([4, 8,12,16, 24, 32, 48, 64, 96, 128]) / 32. * med_pe
        ktile_size_array = np.array([4, 8, 12, 16, 24, 32, 48, 64, 96, 128]) / 32. * med_ktile_size

        action_space = [np.maximum(np.round(pe_array).astype(np.int32), 1),  # Choice for number of PEs
                        np.maximum(np.round(ktile_size_array).astype(np.int32),
                                   1), ]  # Choice for number of buffer unit

        action_bound = [np.max(action_space[0]), np.max(action_space[1])]
        action_bottom = [np.min(action_space[0]), np.min(action_space[1])]
        return action_size, action_space, action_bound, action_bottom

    def get_draw(self, model_name, model_defs):
        if model_name in self.model2draw:
            return self.model2draw[model_name]
        else:
            total_latency = []
            avg_pe = math.floor(self.constraint_value[0] / self.total_step)
            for i in range(len(model_defs)):
                maestro_state = np.concatenate((self.model_defs[i], np.array([avg_pe, 1])))
                reward, mac, constraint = self.oberserve_maestro(maestro_state)
                if not reward or reward is None:
                    return None
                else:
                    total_latency.append(reward[0])
            total_latency = np.array(total_latency)
            draw = total_latency.argsort()
            self.model2draw[model_name] = draw
            return draw

    def get_l1_size_array(self, model_name, model_defs):
        if model_name in self.model2l1_size_array:
            return self.model2l1_size_array[model_name]
        else:
            l1_size_array = np.zeros((self.total_step, 2))
            for i in range(len(model_defs)):
                k, c, x, y, r, s, t = model_defs[i, 0:7]
                if t == 2:
                    base_size = 2 * r * s
                    l1_size_array[i, 0] = 2 * base_size + base_size
                    l1_size_array[i, 1] = 0
                else:
                    if self.dataflow == 'eye':
                        base_size = r + s
                    else:
                        base_size = 2 * r * s
                    l1_size_array[i, 0] = 2 * base_size
                    l1_size_array[i, 1] = base_size

            self.model2l1_size_array[model_name] = l1_size_array
            return l1_size_array

    @property
    def get_state(self):
        """

        """
        return self.state

    def shuffle_model(self):
        draw = np.random.permutation(self.total_step)
        self.model_defs = self.model_defs_saved[draw]
        self.model_defs_norm = self.model_defs_norm_saved[draw]
        self.draw = draw

    def set_fitness(self, fitness="latency"):
        self.fitness = fitness

    def set_constraint(self, constraint="buffer_size"):
        self.constraint = constraint

    def reset(self):
        """

        """
        self.mac_rec = []
        self.sig = 1
        self.sol = []
        self.mode = 0
        self.actions_step = 0
        self.total_used_constraints = np.array([0., 0., 0.])
        dimensions = self.model_defs_norm[self.mode]
        self.action_idx = [0,0]
        self.action = np.array([self.action_space[idx][val] for idx, val in enumerate(self.action_idx)])
        self.state = self.norm_state(np.concatenate(
            (dimensions,
             np.array([0, 0, 0, 0],
                      dtype=np.float))))
        self.whole_eps_rewards = []
        infos = {}

        min_used_l1_size = self.action_space[1][0] * self.l1_size_array[self.mode + 1:self.total_step,
                                                     0].sum() + self.l1_size_array[self.mode + 1:self.total_step,
                                                                1].sum()
        ktile_size_max = (self.constraint_value[1] - min_used_l1_size - self.l1_size_array[self.mode, 1]) / \
                         self.l1_size_array[self.mode, 0]

        pe_constraints = {
            'remained_pes': self.constraint_value[0] - self.action_space[0][0] * (self.total_step - 1),
            'action_space': self.action_space,
            'dim': self.model_defs[self.mode],
            'ktile_size_max': ktile_size_max
        }
        return self.state, infos, pe_constraints


    def norm_mac(self):
        mac_rec = np.array(self.mac_rec)
        impt =  mac_rec/ np.sum(mac_rec)
        return impt


    def update_reward_impt(self, done):
        # impt = np.ones(1, (self.mode) * self.n_action_steps + self.actions_step + 1)
        # if self.fitness == "thrpt_ave":
        #     impt_thrpt = self.norm_mac()
        #     impt[:len(impt_thrpt)] = impt_thrpt
        # return impt
        impt = None
        if self.fitness == "thrpt_ave":
            self.mac_rec.append(self.observation[-1])
            if done:
                impt = self.norm_mac()
        return impt

    def norm_state(self, T):
        T[:-1] = (T[:-1] - 0.5) * 2
        return T

    def update_mode_and_step(self):
        self.actions_step += 2
        if self.actions_step ==self.n_action_steps:
            self.mode+=1
            self.actions_step = 0

    def get_ref_constraint(self, bound):
        sol = [bound[:self.n_action_steps] for i in range(len(self.model_defs))]
        _, total_constraint = self.exterior_search(sol)
        return total_constraint
    def update_best_reward_list(self, succeed):
        self.epoch += 1
        self.reward_whole_eps = self.prev_reward_whole_eps if not succeed else self.reward_whole_eps
        self.prev_reward_whole_eps = self.reward_whole_eps
        self.reward_rec.append(self.reward_whole_eps)
        self.best_rewards.append(self.best_reward_whole)


    def update_reward_whole(self):
        whole_eps_rewards = np.array(self.whole_eps_rewards)
        eps_reward = (whole_eps_rewards[:, 0].sum() + (self.batch_size - 1) * whole_eps_rewards[:, 0].min()) * \
                     whole_eps_rewards[:, 1].sum()
        eps_reward *= -1
        if eps_reward > self.best_reward:
            self.best_reward = eps_reward
            self.sol = self.retrive_sol_order(self.sol)
            self.best_sol = {"epoch":self.epoch,
                            "draw": self.draw,
                             "sol": self.sol,
                             "best_reward": eps_reward,
                             "ctr": self.total_used_constraints}

    def retrive_sol_order(self, sol):
        ordered_sol = [None for _ in range(len(sol))]
        for i, d in enumerate(self.draw):
            ordered_sol[d] = copy.deepcopy(sol[i])
        return ordered_sol

    def get_reward(self, maestro_state):
        table_entry = tuple(maestro_state)
        if table_entry in self.exp_table:
            reward, mac, constraint = self.exp_table[table_entry]
        else:
            ret = self.oberserve_maestro(maestro_state)
            if ret is None:
                print("ret is None")
                return -1, [float("Inf"), float("Inf"), float("Inf")]
            reward, mac, constraint = ret
            self.exp_table[table_entry] = (reward, mac, constraint)
        if not reward:
            self.sig = -1
            print("Not reward")
            return -1, [float("Inf"), float("Inf"), float("Inf")]

        return reward, constraint

    def get_valid_action_range(self):
        valid_action_range = []
        for i, r in enumerate(self.left_resource):
            action_space_this = self.action_space[i]
            valid_action_range.append([1 if val==True else 0 for val in action_space_this<r])
        return np.vstack(valid_action_range).astype(float)

    def update_total_reward_constraint(self, reward, constraint):
        self.whole_eps_rewards.append(reward)
        self.total_used_constraints += constraint

    def is_cluster_step(self):
        if self.n_action_steps >2:
            if self.actions_step == self.n_action_steps -1:
                return True
            self.action[2] = min(self.action[2], self.action[0])
        return False

    def step(self, action):
        infos = {}
        infos["succeed"] = 0
        done = 0
        action = action.cpu().numpy().flatten()
        #===use continuous========================================
        # action = np.clip((action+1)/0.5 * 12, a_min=0, a_max=11)
        #=======================================================
        self.action_idx[:] = action
        action_val = [self.action_space[i][int(a)] for i, a in enumerate(action)]
        self.action[:] = action_val

        if self.model_defs[self.mode, 6] == 2:
            ClusterSz = 1
            k_dim = self.model_defs[self.mode, 1]
            self.action[0] = min(k_dim * ClusterSz, self.action[0])
            self.action[1] = min(math.ceil(k_dim / self.action[0]), k_dim, self.action[1])
        else:
            if self.dataflow == 'dla':
                ClusterSz = 64
                k_dim = self.model_defs[self.mode, 0]
                self.action[0] = min(k_dim * ClusterSz, self.action[0])
                ClusterSz = min(self.action[0], 64)
                self.action[1] = min(math.ceil(k_dim / math.floor(self.action[0] / ClusterSz)), k_dim, self.action[1])
            elif self.dataflow == 'eye':
                ClusterSz = 3
                k_dim = self.model_defs[self.mode, 0]
                y_dim = self.model_defs[self.mode, 2]
                self.action[0] = min(y_dim * ClusterSz, self.action[0])
                self.action[1] = min(k_dim, self.action[1])
            elif self.dataflow == 'shi':
                ClusterSz = 8
                k_dim = self.model_defs[self.mode, 0]
                y_dim = self.model_defs[self.mode, 2]
                self.action[0] = min(y_dim * ClusterSz, self.action[0])
                self.action[1] = min(k_dim, self.action[1])

        maestro_state = np.concatenate((self.model_defs[self.mode],  self.action)).copy()
        reward_saved, constraint = self.get_reward(maestro_state)
        self.update_total_reward_constraint(reward_saved, constraint)
        self.sol.append((copy.deepcopy(self.action)).clip(1))
        whole_eps_rewards = np.array(self.whole_eps_rewards)
        if self.total_used_constraints[0] > self.constraint_value[0] or self.total_used_constraints[1] > \
                self.constraint_value[1]:
            eps_reward = (whole_eps_rewards[:, 0].sum() + (self.batch_size - 1) * whole_eps_rewards[:, 0].min()) * \
                         whole_eps_rewards[:, 1].sum()
            reward = -1 * eps_reward
            print(self.model_name, self.total_used_constraints, self.min_eps_reward, reward)
            done = 1
        else:
            if self.mode == self.total_step - 1:
                eps_reward = (whole_eps_rewards[:, 0].sum() + (self.batch_size - 1) * whole_eps_rewards[:, 0].min()) * \
                             whole_eps_rewards[:, 1].sum()
                eps_reward *= -1
                reward = eps_reward - self.min_eps_reward
                self.min_eps_reward = min(eps_reward, self.min_eps_reward)
                self.model2min_eps_reward[self.model_name] = min(self.model2min_eps_reward[self.model_name],
                                                                 self.min_eps_reward)
            else:
                reward = 0
        if reward == -1:
            done = 1
        self.update_mode_and_step()
        if self.mode < self.total_step:
            dimensions = self.model_defs_norm[self.mode]
            self.state = self.norm_state(np.concatenate(
                (dimensions,
                 np.array([self.mode / self.total_step, *self.action / self.action_bound, self.actions_step],
                          dtype=np.float))))
            min_used_l1_size = self.action_space[1][0] * self.l1_size_array[self.mode + 1:self.total_step,
                                                         0].sum() + self.l1_size_array[self.mode + 1:self.total_step,
                                                                    1].sum()
            ktile_size_max = (self.constraint_value[1] - self.total_used_constraints[1] - min_used_l1_size -
                              self.l1_size_array[self.mode, 1]) / self.l1_size_array[
                                 self.mode, 0]
            pe_constraints = {
                'remained_pes': self.constraint_value[0] - self.total_used_constraints[0] - self.action_space[0][0] * (
                            self.total_step - self.mode - 1),
                'action_space': self.action_space,
                'dim': self.model_defs[self.mode],
                'ktile_size_max': ktile_size_max
            }
        else:
            self.state = None
            pe_constraints = {}
        if self.mode == self.total_step and not done:
            infos["succeed"] = 1
            done = 1
            self.update_reward_whole()
            self.sol_record.append(self.sol)
        impt = self.update_reward_impt(done)
        return self.state, reward, done, infos, self.sig, impt, pe_constraints

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        values = []
        term = self.fitness
        if term =="energy":
            reward =-energy
        elif term == "thrpt_ave":
            reward = throughput
        elif term == "LEP":
            reward = -energy * runtime
        elif term == "LAP":
            reward = -area * runtime
        elif term == "EAP":
            reward = -area * energy
        elif term == "thrpt" or term=="thrpt_naive":
            reward=throughput
        elif term == "thrpt_btnk":
            reward = throughput
        elif term == "latency":
            reward=-runtime
        elif term =="area":
            reward = -area
        elif term == "l1_size":
            reward = - l1_size
        elif term == "l2_size":
            reward = -l2_size
        elif term == "power":
            reward = -power
        else:
            raise NameError('Undefined fitness type')
        values.append(reward)
        return [-runtime, -energy], mac, [l1_size, l2_size]


    def check_constraint(self, actions):
        used = np.sum(actions, axis=0)
        if any(used > self.total_resources[len(used)]):
            return False
        return True
    def ransom_search(self, max_epoch=1000, chpt_file="trial.plt"):
        self.chkpt_file = chpt_file
        n_layer = len(self.model_defs)
        best_reward = 0
        best_sol = None
        best_reward_record = []

        for epoch in range(max_epoch):
            self.epoch = epoch
            guess_action = []
            for _ in range(n_layer):
                pe = 2**random.randint(PE_RANGE_LOG[0], PE_RANGE_LOG[1])
                bw = 2**random.randint(BW_RANGE_LOG[0], BW_RANGE_LOG[1])
                action = [pe, bw]
                guess_action.append(action)
            if not self.check_constraint(guess_action):
                reward = 0
            else:
                reward = self.exterior_search(guess_action)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_sol = guess_action
            print("Epoch {}: reward: {}".format(self.epoch, self.best_reward))
            if self.epoch %100==0:
                self.save_chkpt()

            self.best_rewards.append( self.best_reward)
        return self.best_rewards, self.best_sol

    def dfs(self, left_layers, guess_action):
        if left_layers == 0:
            self.epoch +=1
            if self.check_constraint(guess_action):
                reward = self.exterior_search(guess_action)
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_sol = guess_action
                print("Epoch {}: reward: {}".format(self.epoch, self.best_reward))
            self.best_rewards.append( self.best_reward)

            if self.epoch %100==0:
                self.save_chkpt()
            return
        for pe in [2**i for i in range(PE_RANGE_LOG[0], PE_RANGE_LOG[1]+1)]:
            for bw in [2**i for i in range(BW_RANGE_LOG[0], BW_RANGE_LOG[1]+1)]:
                action=[pe, bw]
                guess_action.append(action)
                self.dfs(left_layers-1, guess_action)
                guess_action.pop()
    def exhasustive_search(self, chkpt_file="trial.pgt"):
        n_layer = len(self.model_defs)
        self.best_reward = 0
        best_sol = None
        best_reward_record = []
        guess_action = []
        self.best_rewards = []
        self.epoch = 0
        self.best_sol = None
        self.chkpt_file = chkpt_file
        self.dfs(n_layer, guess_action)
        return self.best_rewards, self.best_sol

    def load_chkpt(self, chkpt):
        self.reward_rec = chkpt["reward_rec"]
        self.best_reward = chkpt["best_reward"]
        self.best_rewards= chkpt["best_rewards"]
        self.best_sol= chkpt["best_sol"]
        self.worst_reward = chkpt["worst_reward"]

    def get_chkpt(self):
        return {
            "best_sol": self.best_sol,
            "best_reward": self.best_reward
        }
    def save_chkpt(self, chkpt_file=None):
        if chkpt_file is None:
            chkpt_file = self.chkpt_file
        chkpt = self.get_chkpt()
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)

    def exterior_search_special(self, actions, action_size=2):
        total_reward = None
        mac_rec_list = []
        latency_list = []
        total_constraint = 0
        for i in range(len(actions)):
            action = actions[i]
            maestro_state = np.concatenate((self.model_defs[i], action))
            reward, constraint = self.oberserve_maestro(maestro_state)
            if reward == None:
                return None
            else:
                mac_rec_list.append(self.observation[-1])
                latency_list.append(self.observation[0])
            total_constraint += constraint
        total_reward = sum(mac_rec_list)/sum(latency_list)
        return total_reward, total_constraint

    def quick_observe(self, maestro_state):
        table_entry = tuple(maestro_state)
        if table_entry in self.exp_table:
            reward, constraint = self.exp_table[table_entry]
        else:
            ret = self.oberserve_maestro(maestro_state)
            if ret is None:
                return None, float("Inf")
            reward, constraint = ret
            self.exp_table[table_entry] = (reward, constraint)
        return reward, constraint

    def exterior_search(self, actions, action_size=2):
        if self.fitness == "thrpt_ave" or self.fitness=="thrpt_naive":
            return self.exterior_search_special(actions, action_size)
        total_mac = []
        total_latency = []
        total_energy = []
        total_constraint = np.array([0., 0., 0.])
        min_reward = float("Inf")
        if len(actions) > len(self.model_defs):
            print(len(actions), len(self.model_defs))
        for i in range(len(actions)):
            action = actions[i]
            maestro_state = np.concatenate((self.model_defs_saved[i], action))
            reward, mac, constraint = self.oberserve_maestro(maestro_state)
            if not reward :
                return None, float("Inf")
            total_latency.append(reward[0])
            total_energy.append(reward[1])
            total_mac.append(-1 * mac)
            total_constraint = total_constraint + constraint
        total_latency = np.array(total_latency)
        total_energy = np.array(total_energy)
        total_mac = np.array(total_mac)
        total_lep = -1 * total_latency.sum() * total_energy.sum()
        total_latency = (total_latency.sum() + (self.batch_size-1)*total_latency.min())
        total_energy = total_energy.sum()
        return total_latency, total_energy, total_lep, total_constraint

    def write_maestro(self, dimension, dataflow="dla", KTileSz=1, CTileSz=1, ClusterSz=4, m_file=None, layer_id=0,df_idx=None):
        if df_idx is not None:
            dataflow = df_dict[df_idx]
        if len(dimension) > 6:
            m_type = m_type_dicts[int(dimension[-1])]
        else:
            m_type = "CONV"
        with open("../../data/dataflow/{}.m".format(dataflow), "r") as fd:
            with open("../../data/dataflow/dpt.m", "r") as fdpt:
                with open("{}.m".format(m_file), "w") as fo:
                    fo.write("Constant KTileSz {};\n".format(KTileSz))
                    fo.write("Constant CTileSz {};\n".format(CTileSz))
                    fo.write("Constant ClusterSz {};\n".format(ClusterSz))
                    fo.write("Network {} {{\n".format(layer_id))
                    fo.write("Layer {} {{\n".format(m_type))
                    fo.write("Type: {}\n".format(m_type))
                    fo.write(
                        "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                            *dimension))
                    if m_type == "CONV" or m_type == "TRCONV":
                        fd.seek(0)
                        fo.write(fd.read())
                    else:
                        fdpt.seek(0)
                        fo.write(fdpt.read())
                    fo.write("}\n")
                    fo.write("}")

    def oberserve_maestro(self, state, firsttime=False):
        m_file = self.random_file_name
        dimension = state[:self.dim_size]

        actions =  state[-self.n_action_steps:]
        if self.n_action_steps==2:
            num_pe, KTileSz = actions.astype(int).squeeze()
            if self.dataflow == 'dla':
                ClusterSz = min(num_pe, 64)
            elif self.dataflow == 'eye':
                ClusterSz = min(num_pe, 3)
            elif self.dataflow == 'shi':
                ClusterSz = min(num_pe, 8)
            self.resource_state = [num_pe, KTileSz]
            if self.is_gemm:
                self.write_maestro_gemm(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz, m_file=m_file)
            else:
                self.write_maestro(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz, m_file=m_file)
        else:
            num_pe, KTileSz, df_idx = actions.astype(int).squeeze()
            self.resource_state = [num_pe, KTileSz]
            if self.dataflow == 'dla':
                ClusterSz = min(num_pe, 64)
            elif self.dataflow == 'eye':
                ClusterSz = min(num_pe, 3)
            elif self.dataflow == 'shi':
                ClusterSz = min(num_pe, 8)
            if self.is_gemm:
                self.write_maestro_gemm(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz,
                                        m_file=m_file, df_idx=df_idx)
            else:
                self.write_maestro(dimension, dataflow=self.dataflow, KTileSz=KTileSz, ClusterSz=ClusterSz,
                                   m_file=m_file, df_idx=df_idx)

        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw=81920000",
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(num_pe),
                   "--num_simd_lanes=1", "--l1_size=819200000",
                   "--l2_size=819200000", "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]
            reward, mac, constraint = self.judge()
            return reward, mac, np.array([num_pe, constraint[0], constraint[1]])
        except Exception as e:
            print(e)
            print("+"*20)
            print(num_pe, KTileSz, ClusterSz)
            print("+" * 20)
            return None

    def write_maestro_gemm(self, dimension, dataflow="dla", KTileSz=1, CTileSz=1, ClusterSz=4, m_file=None, layer_id=0,df_idx=None):
        if df_idx is not None:
            dataflow = df_dict[df_idx]
        m_type = "CONV"
        SzM, SzN, SzK = dimension
        dimension = [SzN, SzK, SzM, 1, 1, 1]
        with open("{}_f.m".format(dataflow), "r") as fd:
            with open("dpt_f.m", "r") as fdpt:
                with open("{}.m".format(m_file), "w") as fo:
                    fo.write("Constant KTileSz {};\n".format(KTileSz))
                    fo.write("Constant CTileSz {};\n".format(CTileSz))
                    fo.write("Constant ClusterSz {};\n".format(ClusterSz))
                    fo.write("Network {} {{\n".format(layer_id))
                    fo.write("Layer {} {{\n".format("CONV"))
                    fo.write("Type: {}\n".format(m_type))
                    fo.write(
                        "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                            *dimension))
                    if m_type == "CONV" or m_type == "TRCONV":
                        fd.seek(0)
                        fo.write(fd.read())
                    else:
                        fdpt.seek(0)
                        fo.write(fdpt.read())
                    fo.write("}\n")
                    fo.write("}")


