import random
import time

import gym
import numpy as np
import redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.misc import imresize
from torch.autograd import Variable

from cnslib.agent import Agent
from cnslib.argtypes import list_of
from cnslib.genepool import GenePool
from cnslib.genome import ModelGenome


class MLP(nn.Module):
    def __init__(self, input_shape, base_filters, num_hidden, num_actions):
        super(MLP, self).__init__()
        num_input = int(np.prod(input_shape))
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], base_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(base_filters * 2, base_filters * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(base_filters * 4, base_filters * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 8 * 16, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_actions),
            nn.Softmax()
        )

    def forward(self, x):
        z = self.convs(x)
        z = z.view(z.size(0), -1)
        return self.classifier(z)


def main(config):
    environment = gym.make(config.env)
    state_shape = (3, 64, 64)
    num_hidden = config.num_hidden
    num_actions = environment.action_space.n
    base_filters = 8
    agent = Agent(MLP(state_shape, base_filters, num_hidden, num_actions))
    best_agent = Agent(MLP(state_shape, base_filters, num_hidden, num_actions))
    agent.randomize(config.gene_weight_ratio, config.freq_weight_ratio, config.v_init)
    agent.update_model()
    print(agent.summary())
    genepool = GenePool()
    num_episodes = 0
    while True:
        print('Starting episode {}'.format(num_episodes))
        best_genomes = genepool.top_n(config.num_best)
        if not config.best:
            reward, steps = run_episode(agent, environment, config)
            print('Reward {} in {} steps'.format(reward, steps))
            genepool.report_score(agent.genome, reward)
            update_agent(agent, reward, best_genomes, config)
        if best_genomes and np.random.uniform() < 0.1 or config.best:
            best_genome, _ = best_genomes[0]#random.choice(best_genomes)
            best_agent.load_genome(best_genome)
            best_agent.update_model()
            best_reward, steps = run_episode(best_agent, environment, config)
            if not config.best:
                genepool.report_score(best_agent.genome, best_reward)

        num_episodes += 1


def update_agent(agent, reward, best_genomes, config):
    if len(best_genomes) >= config.min_genepool:
        _, best_score = best_genomes[0]
        _, worst_best_score = best_genomes[-1]
        print('Genepool top: {}, {}'.format(best_score, worst_best_score))
        if reward < worst_best_score:
            # Our score isn't notable
            agent.crossover(best_genomes)
        else:
            # New high-ish score
            print('new ok score')
    agent.mutate(index_sigma=config.i_sigma, value_sigma=config.v_sigma)
    agent.update_model()
    return best_genomes


def run_episode(agent, environment, config):
    total_reward = 0.
    num_steps = 0
    observation = environment.reset()
    done = False
    while not done:
        if config.render:
            environment.render()
        observation = imresize(observation, (64, 64)).transpose(2, 0, 1)
        action = agent.policy(observation)
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        num_steps += 1
    return total_reward, num_steps


if __name__ == '__main__':
    import argparse
    import multiprocessing

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', default='Pong-v0')
    argparser.add_argument('--num-agents', type=int, default=10)
    argparser.add_argument('--min-genepool', type=int, default=2)
    argparser.add_argument('--num-best', type=int, default=20)
    argparser.add_argument('--render', action='store_true')
    argparser.add_argument('--clear-store', action='store_true')
    argparser.add_argument('--gene-weight-ratio', type=float, default=0.005)
    argparser.add_argument('--freq-weight-ratio', type=float, default=1.)
    argparser.add_argument('--i-sigma', type=float, default=0.5)
    argparser.add_argument('--v-sigma', type=list_of(float), default=1.)
    argparser.add_argument('--v-init', type=list_of(float), default=(-1., 1.))
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--best', action='store_true')
    config = argparser.parse_args()

    genepool = GenePool()
    if config.clear_store:
        genepool.clear()

    if config.best:
        main(config)
    else:
        processes = []
        for _ in range(config.num_agents):
            p = multiprocessing.Process(target=main, args=(config,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
