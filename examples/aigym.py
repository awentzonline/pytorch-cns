import random
import time

import gym
import numpy as np
import redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from cnslib.agent import Agent
from cnslib.argtypes import list_of
from cnslib.genepool import GenePool
from cnslib.genome import ModelGenome


class MLP(nn.Module):
    def __init__(self, num_input, num_hidden, num_actions):
        super(MLP, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_actions),
            nn.Softmax()
        )

    def forward(self, x):
        return self.main(x)


def main(config):
    environment = gym.make(config.env)
    state_shape = environment.observation_space.low.shape
    num_hidden = config.num_hidden
    num_actions = environment.action_space.n
    model = MLP(state_shape[0], num_hidden, num_actions)
    agent = Agent(model)
    agent.randomize(config.gene_weight_ratio, config.freq_weight_ratio, config.v_init)
    print(agent.genome)
    genepool = GenePool()
    if config.clear_store:
        genepool.clear()
    num_episodes = 0
    while True:
        print('Starting episode {}'.format(num_episodes))
        reward, steps = run_episode(agent, environment, config)
        print('Reward {} in {} steps'.format(reward, steps))
        best_genomes = genepool.top_n(config.num_best)
        # show off genome
        if (num_episodes + 1) % 50 == 0 and config.render:
            print('******** EXHIBITION ***********')
            print(agent.genome)
            best_genome, _ = best_genomes[0]
            agent.load_genome(best_genome)
            run_episode(agent, environment, config)

        if len(best_genomes) < config.min_genepool:
            genepool.report_score(agent.genome, reward)  # we're still gathering scores
        else:
            _, best_score = best_genomes[0]
            _, worst_best_score = best_genomes[-1]
            print('Genepool top: {}, {}'.format(best_score, worst_best_score))
            if reward < worst_best_score:
                # Our score isn't notable
                agent.crossover(best_genomes)
            else:
                # New high-ish score
                print('new ok score')
                genepool.report_score(agent.genome, reward)
        agent.mutate()
        agent.update_model()
        num_episodes += 1


def run_episode(agent, environment, config):
    total_reward = 0.
    num_steps = 0
    observation = environment.reset()
    done = False
    while not done:
        if config.render:
            environment.render()
        action = agent.policy(observation)
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        num_steps += 1
    return total_reward, num_steps


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', default='CartPole-v0')
    argparser.add_argument('--num-agents', type=int, default=10)
    argparser.add_argument('--min-genepool', type=int, default=5)
    argparser.add_argument('--num-best', type=int, default=20)
    argparser.add_argument('--render', action='store_true')
    argparser.add_argument('--clear-store', action='store_true')
    argparser.add_argument('--gene-weight-ratio', type=float, default=0.05)
    argparser.add_argument('--freq-weight-ratio', type=float, default=1.)
    argparser.add_argument('--v-change', type=list_of(float), default=(-1., 1.))
    argparser.add_argument('--v-init', type=list_of(float), default=(-1., 1.))
    argparser.add_argument('--num-hidden', type=int, default=32)
    config = argparser.parse_args()
    main(config)
