'''
Working version for both GEMM and nonGEMM
'''
import copy
import random
import os, sys
import argparse
import torch
import numpy as np
from collections import deque

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from env_confx import MaestroEnvironment
import pickle
from agent import Agent
from model import Actor
import torch.optim as optim
import pandas as pd
from datetime import datetime
from train_utils import *
import glob


def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

def adjust_lr(ratio, actor_optimizer, min_lr=1e-8):
    for param_group in actor_optimizer.param_groups:
        param_group['lr'] = max(min_lr, param_group['lr'] * ratio)


def train(n_episodes=100000, max_t=1000, print_every=10, eps=0., temperature=1.):
    actor = Actor(dim_size=dim_size, resource_size=2, n_action_steps=2, action_size=action_size, h_size=128, num_steps=opt.num_steps).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    scheduler =optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, factor=0.9, min_lr=1e-6)

    agent = Agent(actor, h_size=128, device=device)
    agent.set_fitness(opt.fitness)
    agent.reset()

    train_env = None
    best_reward_whole = -2**30

    for i_episode in range(0, n_episodes):
        if (i_episode+1) % 100 ==0:
            eps /= 1.2
            temperature /=1.01
            temperature = max(temperature,1)
            adjust_lr(ratio=0.8, actor_optimizer=actor_optimizer, min_lr=1e-6)

        # sample CNN
        if i_episode % len(models) == 0:
            random.shuffle(models)
            log_str = "Start {}th Training {} !!!\n".format(i_episode, models)
            epf.write(log_str)
            epf.flush()
            print(log_str)
        for train_model in models:
            train_m_file = os.path.join(m_file_path, train_model + ".csv")
            train_df = pd.read_csv(train_m_file)
            train_model_defs = train_df.to_numpy()

            if train_env is None:
                train_env = MaestroEnvironment(model_name=train_model, model_defs=train_model_defs,dim_size=dim_size,
                                               resource_size=2, n_action_steps=2,
                                               fitness=opt.fitness, dataflow=opt.df,
                                               num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size)
            else:
                train_env.reset_dimension(model_name=train_model, model_defs=train_model_defs, dim_size=dim_size,
                                          resource_size=2, n_action_steps=2,
                                          fitness=opt.fitness, dataflow=opt.df,
                                          num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size)

            train_env.set_fitness(opt.fitness)
            train_env.set_constraint(opt.cstr)

            theta = {}
            for name, param in actor.named_parameters():
                # theta[name] = param.clone().detach().requires_grad_(True)
                theta[name] = param
            for i_adapt in range(ADAPT_STEPS):
                score = 0
                state, infos, pe_constraints = train_env.reset()
                for _ in range(max_t):
                    action, log_prob = agent.act(state, infos, eps, temperature, theta, pe_constraints)
                    next_state, reward, done, infos, sig, impt, pe_constraints = train_env.step(action)
                    agent.step(state, action, log_prob, reward, next_state, done, sig, impt, infos)
                    state = next_state
                    score += reward
                    if done:
                        break

                # inner-gradient
                learned_loss = compute_learned_loss(agent, theta)
                agent.reset_learned_input()
                if learned_loss is None:
                    continue
                inner_gradient = torch.autograd.grad(
                    learned_loss,
                    [v for _, v in theta.items()],
                    create_graph=not first_order,
                    retain_graph=not first_order,
                    allow_unused=True,
                )
                theta = SGD_step(theta, inner_gradient, INNER_LR)


            score = 0
            state, infos, pe_constraints = train_env.reset()
            for _ in range(max_t):
                action, log_prob = agent.act(state, infos, eps,temperature, theta, pe_constraints)
                next_state, reward, done, infos, sig, impt, pe_constraints = train_env.step(action)
                agent.step(state, action, log_prob, reward, next_state, done, sig, impt, infos)
                state = next_state
                score += reward
                if done:
                    break

            loss = compute_loss(agent, GAMMA, impt, infos)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), CLIPPING_MODEL)
            torch.nn.utils.clip_grad_norm_(actor.lstm.parameters(), CLIPPING_LSTM)
            actor_optimizer.step()

            agent.reset()
            for f in glob.glob("*.m"):
                os.remove(f)
            for f in glob.glob("*.csv"):
                os.remove(f)

        if (i_episode + 1) % len(models) == 0:
            torch.save(actor.state_dict(), os.path.join(outdir_exp, 'actor_epoch_{}_chkpt.pth'.format(i_episode)))

        if (i_episode + 1) % 100 == 0:
            val(models, i_episode, n_episodes=test_epochs, max_t=max_t, print_every=10, eps=0, temperature=1,
                actor_state_dict=actor.state_dict())
            val(test_models, i_episode, n_episodes=test_epochs, max_t=max_t, print_every=10, eps=0, temperature=1,
                actor_state_dict=actor.state_dict())
            chkpt_file = os.path.join(outdir_exp, 'result_epoch_{}.plt'.format(i_episode))
            with open(chkpt_file, "wb") as fd:
                pickle.dump(model2sol, fd)
        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)


def val(train_models, i_episode, n_episodes=100000, max_t=1000, print_every=10, eps=0, temperature=1, actor_state_dict=None):
    log_str = "Start Validating !!!\n"
    epf.write(log_str)
    epf.flush()
    print(log_str)
    train_env = None

    for train_model in train_models:
        start = datetime.now()
        start_date = "{}".format(start.date())
        start_time = "{}".format(start.time())
        val_actor = Actor(dim_size=dim_size, resource_size=2, n_action_steps=2, action_size=action_size, h_size=128,
                          num_steps=opt.num_steps).to(device)
        val_actor.load_state_dict(actor_state_dict)

        val_agent = Agent(val_actor, h_size=128, device=device)
        val_agent.set_fitness(opt.fitness)
        val_agent.reset()
        best_score = float("-Inf")
        best_reward_whole = float("-Inf")
        best_latency_energy_lep_whole = [float("-Inf"), float("-Inf"), float("-Inf")]
        best_constraint = [0, 0, 0]
        best_episode = 0
        best_sol_whole = None
        scores_window = deque(maxlen=print_every)
        scores = []
        episodes = 0
        has_succeed_history = False
        train_m_file = os.path.join(m_file_path, train_model + ".csv")
        train_df = pd.read_csv(train_m_file)
        train_model_defs = train_df.to_numpy()

        if train_env is None:
            train_env = MaestroEnvironment(model_name=train_model, model_defs=train_model_defs, dim_size=dim_size,
                                           resource_size=2, n_action_steps=2,
                                           fitness=opt.fitness, dataflow=opt.df,
                                           num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size)
        else:
            train_env.reset_dimension(model_name=train_model, model_defs=train_model_defs, dim_size=dim_size,
                                      resource_size=2, n_action_steps=2,
                                      fitness=opt.fitness, dataflow=opt.df,
                                      num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size)
        train_env.set_fitness(opt.fitness)
        train_env.set_constraint(opt.cstr)

        theta = {}
        for name, param in val_actor.named_parameters():
            # theta[name] = param.clone().detach().requires_grad_(True)
            theta[name] = param
        for i_adapt in range(n_episodes):
            score = 0
            state, infos, pe_constraints = train_env.reset()
            for _ in range(max_t):
                action, log_prob = val_agent.act(state, infos, eps,temperature, theta, pe_constraints)
                next_state, reward, done, infos, sig, impt, pe_constraints = train_env.step(action)
                val_agent.step(state, action, log_prob, reward, next_state, done, sig, impt, infos)
                state = next_state
                score += reward
                if done:
                    break

            # inner-gradient
            learned_loss = compute_learned_loss(val_agent, theta)
            if learned_loss is None:
                continue

            inner_gradient = torch.autograd.grad(
                learned_loss,
                [v for _, v in theta.items()],
                create_graph=not first_order,
                retain_graph=not first_order,
                allow_unused=True,
            )

            theta = SGD_step(theta, inner_gradient, INNER_LR)

            env_chkpt = train_env.get_chkpt()
            if env_chkpt["best_sol"] is not None:
                best_sol = env_chkpt["best_sol"]
                total_latency, total_energy, total_lep, episode_constraint = train_env.exterior_search(best_sol["sol"])
                if -1 * total_latency * total_energy > best_reward_whole:
                    best_reward_whole = -1 * total_latency * total_energy
                    best_latency_energy_lep_whole[0] = total_latency
                    best_latency_energy_lep_whole[1] = total_energy
                    best_latency_energy_lep_whole[2] = total_lep
                    best_constraint = episode_constraint
                    best_episode = i_adapt
                    best_sol_whole = best_sol

        val_agent.reset()
        log_str = f'RL result: {train_model} at {best_episode} reward: {best_reward_whole:.4e},' \
                  f'latency: {best_latency_energy_lep_whole[0]}, energy: {best_latency_energy_lep_whole[1]},' \
                  f'lep: {best_latency_energy_lep_whole[2]}, Used PEs: {best_constraint[0]}, ' \
                  f'Used SLs: {best_constraint[1]}, Used SG: {best_constraint[2]}\n'
        print(log_str)
        epf.write(log_str)
        epf.flush()

        end = datetime.now()
        end_date = "{}".format(end.date())
        end_time = "{}".format(end.time())
        log_str = 'Running time: from {} to {}\n'.format(start_time, end_time)
        print(log_str)
        epf.write(log_str)
        epf.flush()

        if model2sol[train_model]['best_reward'] is None or best_reward_whole > model2sol[train_model]['best_reward']:
            model2sol[train_model]['best_reward'] = best_reward_whole
            model2sol[train_model]['best_latency'] = best_latency_energy_lep_whole[0]
            model2sol[train_model]['best_energy'] = best_latency_energy_lep_whole[1]
            model2sol[train_model]['best_lep'] = best_latency_energy_lep_whole[2]
            model2sol[train_model]['best_sol'] = best_sol_whole
            model2sol[train_model]['best_constraint'] = best_constraint
        model2sol[train_model]['reward_{}'.format(i_episode)] = best_reward_whole
        model2sol[train_model]['sol_{}'.format(i_episode)] = best_sol_whole
        # print(train_env.model2med_l1_size)

        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--fitness', type=str, default="latency", help='The objective.')
    parser.add_argument('--cstr', type=str, default="area", help='The constraint.')
    parser.add_argument('--epochs', type=int, default=500, help='pickle file name')
    parser.add_argument('--test_epochs', type=int, default=100)
    parser.add_argument('--gpu', default=0, type=int, help='which gpu')
    parser.add_argument('--df', default="shi", type=str, help='The dataflow strategy.')
    parser.add_argument('--num_steps', type=int, default=6, help='The number of past policies')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_pe', type=int, default=4096)
    parser.add_argument('--l1_size', type=int, default=8000)
    parser.add_argument('--l2_size', type=int, default=8000)
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tau=1.00
    LR_ACTOR = 1e-3 # learning rate of the actor
    GAMMA = 0.99  # discount factor
    CLIPPING_LSTM = 10
    CLIPPING_MODEL = 100
    INNER_LR = 1e-2
    ADAPT_STEPS = 3

    outdir = opt.outdir
    outdir = os.path.join("../../", outdir)

    first_order= False
    test_models = ['transformer', 'mobilenet_v2', 'resnet50', 'unet']

    action_size = [10, 10]
    dim_size = 7
    batch_size = 512

    exp_name = "Adapt_{}_{}_energy_{}_PE-{}_L1-{}_l2-{}".format(
        opt.df, opt.fitness, opt.cstr, opt.num_pe, opt.l1_size, opt.l2_size)
    outdir_exp = os.path.join(outdir, exp_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_exp, exist_ok=True)

    episode_file = os.path.join(outdir_exp, 'log')
    epf = open(episode_file, 'a')

    test_epochs = opt.test_epochs
    model2sol = {}
    try:
        models = ['googlenet', 'resnet18', 'shufflenet_v2', 'vgg16', 'mnasnet']
        for model in models:
            model2sol[model] = {'best_reward': None,
                                'best_sol': None}
        for model in test_models:
            model2sol[model] = {'best_reward': None,
                                'best_sol': None}
        m_file_path = "../../data/model/"

        # ============================Do training============================================================================================
        set_seed(opt.seed)
        train(n_episodes=opt.epochs, eps=0.0, temperature=1)

    finally:
        for f in glob.glob("*.m"):
            os.remove(f)
        for f in glob.glob("*.csv"):
            os.remove(f)