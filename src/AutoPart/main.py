'''
Working version for both GEMM and nonGEMM
'''
import random
import os, sys
import argparse
import math
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from env_confx import MaestroEnvironment
import pickle
from rl_confx import Agent
import pandas as pd
import copy
from datetime import datetime
import glob
import pdb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def policy_graident(n_episodes=100000, max_t=1000, print_every=10, outfile="out.plt", chkpt_file="trial.plt", eps=0,temperature=1):
    best_score = -2**20
    best_reward_whole = -2**30
    scores_window = deque(maxlen=print_every)
    scores = []
    episodes = 0
    has_succeed_history = False
    rewards = []
    for i_episode in range(1 + episodes, n_episodes + episodes + 1):
        if (i_episode+1) %100 ==0  and has_succeed_history:
            eps /= 1.2
            temperature /=1.01
            temperature = max(temperature,1)
            agent.ajust_lr(ratio=0.8, min_lr=1e-6)

        score = 0
        state, infos, pe_constraints = env.reset()

        for t in range(max_t):
            action, log_prob = agent.act(state, infos, eps, temperature, pe_constraints)
            next_state, reward, done, infos, sig, impt, pe_constraints = env.step(action)
            agent.step(state, action, log_prob, reward, next_state, done, sig, impt, infos)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        if np.mean(scores_window) > best_score:
            best_score = np.mean(scores_window)
        agent_chkpt = agent.get_chkpt()
        env_chkpt = env.get_chkpt()
        others_chkpt =  {
                 "scores": scores,
                 "best_score": best_score,
                 "scores_window":scores_window,
                 "episodes":i_episode}
        chkpt = {"agent_chkpt": agent_chkpt,
                 "env_chkpt": env_chkpt,
                 "others_chkpt": others_chkpt}
        if i_episode % 1 == 0:
            env.save_chkpt(chkpt_file)
            torch.save(chkpt, outfile)

        if infos["succeed"]:
            has_succeed_history = True
            log_str = "Episode {}: succeed\n".format(i_episode)
        else:
            log_str = "Episode {}: finding\n".format(i_episode)
        
        print(log_str)
        emf.write(log_str)
        emf.flush()

        if env_chkpt["best_sol"] is not None:
            best_sol = env_chkpt["best_sol"]
            total_latency, total_energy, total_lep, total_constraint = env.exterior_search(best_sol["sol"])
            log_str = f'model: {model}, latency: {total_latency}, energy: {total_energy},' \
                      f'lep: {total_lep}, Used PEs: {total_constraint[0]}, ' \
                      f'Used SLs: {total_constraint[1]}, Used SG: {total_constraint[2]}\n'
            print(log_str)
            emf.write(log_str)
            emf.flush()
        else:
            total_latency = float('-Inf')
            total_energy = float('-Inf')
            total_lep = float('-Inf')
            total_constraint = [float('-Inf'), float('-Inf'), float('-Inf')]

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--fitness', type=str, default="latency", help='The objective.')
    parser.add_argument('--cstr', type=str, default="area", help='The constraint.')
    parser.add_argument('--epochs', type=int, default=500, help='pickle file name')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu')
    parser.add_argument('--df', default="shi", type=str, help='The dataflow strategy.')
    parser.add_argument('--n_act', type=int, default=2, help='The number of action to make for each layer', choices=[1, 2, 3])
    parser.add_argument('--num_pe', type=int, default=4096)
    parser.add_argument('--l1_size', type=int, default=8000)
    parser.add_argument('--l2_size', type=int, default=8000)
    opt = parser.parse_args()
    device = 'cuda:' + str(opt.gpu) if torch.cuda.is_available() else 'cpu'

    n_acts = opt.n_act
    outdir = opt.outdir
    outdir = os.path.join("../../", outdir)

    m_file_path = "../../data/model/"
    models = ['googlenet', 'resnet18', 'shufflenet_v2', 'vgg16', 'mnasnet']
    test_models = ['transformer', 'resnet50', 'unet', 'mobilenet_v2']
    expLog_file = "./log"
    epf = open(expLog_file, 'w')

    for model in test_models + models:
        m_file = os.path.join(m_file_path, model + ".csv")
        df = pd.read_csv(m_file)
        model_defs = df.to_numpy()
        _, dim_size = model_defs.shape
        try:
            set_seed(42)
            env = MaestroEnvironment(model_name=model, model_defs=model_defs,
                                     dim_size=dim_size, resource_size=2,n_action_steps=2,
                                     fitness=opt.fitness, dataflow=opt.df,
                                     num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size)
            agent = Agent(dim_size=dim_size, resource_size=2, n_action_steps = 2,
                          action_size=env.action_size)
            env.set_fitness(opt.fitness)
            env.set_constraint(opt.cstr)
            exp_name = "Auto_{}_{}_{}_energy_{}_PE-{}_L1-{}_L2".format(
                opt.df, model, opt.fitness, opt.cstr,
                opt.num_pe, opt.l1_size, opt.l2_size)

            outdir_exp = os.path.join(outdir, exp_name)
            os.makedirs(outdir, exist_ok=True)
            os.makedirs(outdir_exp, exist_ok=True)
            chkpt_file_t = os.path.join(outdir_exp, "{}".format("result"))

            outfile = chkpt_file_t + "_o.plt"
            chkpt_file = chkpt_file_t + "_c.plt"
            img_file = chkpt_file_t + ".png"
            log_file = chkpt_file_t + ".csv"
            episode_model_file = chkpt_file_t + ".log"
            emf = open(episode_model_file, 'w')

            agent.set_fitness(opt.fitness)
            agent.reset()
            scores = policy_graident(n_episodes=opt.epochs,  outfile=outfile, chkpt_file=chkpt_file, eps=0.0, temperature=1)
            with open(chkpt_file, "rb") as fd:
                chkpt = pickle.load(fd)

            best_sol = chkpt["best_sol"]
            total_latency, total_energy, total_lep, total_constraint = env.exterior_search(best_sol["sol"])

            log_str = f'model: {model}, latency: {total_latency}, energy: {total_energy},' \
                      f'lep: {total_lep}, Used PEs: {total_constraint[0]}, ' \
                      f'Used SLs: {total_constraint[1]}, Used SG: {total_constraint[2]}\n'
            print(log_str)
            emf.write(log_str)
            emf.flush()
        finally:
            for f in glob.glob("*.m"):
                os.remove(f)
            for f in glob.glob("*.csv"):
                os.remove(f)
