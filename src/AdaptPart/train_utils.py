import math
import numpy as np
import torch


def compute_learned_loss(agent, params=None):
    learned_loss = agent.actor.learned_loss(agent.get_learned_input(), params)
    return learned_loss


def get_params(shared_model, device=None):
    """ Copies the parameters from shared_model into theta. """
    theta = {}
    for name, param in shared_model.named_parameters():
        # Clone and detach.
        # param_copied = param
        param_copied = param.clone().detach().requires_grad_(True)
        if device is not None:
            # theta[name] = torch.tensor(
            #     param_copied,
            #     requires_grad=True,
            #     device=torch.device("cuda:{}".format(gpu_id)),
            # )
            # Changed for pythorch 0.4.1.
            theta[name] = param_copied.to(device)
        else:
            theta[name] = param_copied
    return theta


def SGD_step(theta, grad, lr, testing=False):
    theta_i = {}
    j = 0
    for name, param in theta.items():
        if grad[j] is not None and "exclude" not in name and "ll" not in name:
            theta_i[name] = param - lr * grad[j]
        else:
            theta_i[name] = param

        j += 1

    return theta_i


def compute_loss(agent, gamma, impt, infos):
    EPISIOLON = 2 ** (-12)
    rewards = np.array(agent.rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + EPISIOLON)
    rewards = agent.impt_adj_reward(rewards, impt)
    if agent.fitness == "thrpt_btnk":
        rewards = agent.minmax_reward(rewards)
    dis_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        dis_rewards.insert(0, R)
    dis_rewards = np.array(dis_rewards)
    dis_rewards = (dis_rewards - dis_rewards.mean()) / (dis_rewards.std() + EPISIOLON)
    # dis_rewards = dis_rewards - dis_rewards.mean()
    policy_loss = []
    for log_prob, r in zip(agent.saved_log_probs, dis_rewards):
        policy_loss.append(-log_prob * r)
    policy_loss = torch.cat(policy_loss).sum()

    return policy_loss


def transfer_gradient_to_shared(gradient, shared_model, device):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    i = 0
    for name, param in shared_model.named_parameters():
        if param.requires_grad:
            if gradient[i] is None:
                param._grad = torch.zeros(param.shape).to(device)
            else:
                param._grad = gradient[i]
        i += 1

