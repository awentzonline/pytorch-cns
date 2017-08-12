import random
import time

import gym
import numpy as np
import redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.misc import imresize, imsave
from torch.autograd import Variable

from cnslib.agent import Agent
from cnslib.argtypes import list_of
from cnslib.genepool import GenePool
from cnslib.genome import ModelGenome
from cnslib.optimizers.genepool_async import Optimizer
from cnslib.scoreboard import AgentScoreboard


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 1.0)


class CNN(nn.Module):
    def __init__(self, input_shape, base_filters, num_hidden, num_actions):
        super(CNN, self).__init__()
        num_input = int(np.prod(input_shape))
        self.num_hidden = num_hidden
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], base_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(base_filters, base_filters * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(base_filters * 2, base_filters * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 9 x 9
        )
        # for p in self.convs.parameters():
        #     p.requires_grad = False  # use random conv features
        #self.convs.apply(weights_init)
        self.conv_out_size = base_filters * 2 * 6 * 6
        self.rnn = nn.RNN(self.conv_out_size, self.num_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden, num_actions),
            nn.Softmax()
        )

    def forward(self, x, hidden):
        z = self.convs(x)
        z = z.view(z.size(0), 1, -1)
        z, hidden = self.rnn(z, hidden)
        return self.classifier(z.view(z.size(0), -1)), hidden

    def init_hidden(self):
        return Variable(torch.randn(1, 1, self.num_hidden))


class MLP(nn.Module):
    def __init__(self, input_shape, base_filters, num_hidden, num_actions):
        super(MLP, self).__init__()
        num_input = int(np.prod(input_shape))
        self.num_hidden = num_hidden
        self.rnn = nn.RNN(num_input, self.num_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden, num_actions),
            nn.Softmax()
        )

    def forward(self, x, hidden):
        x = x.view(x.size(0), 1, -1)
        z, hidden = self.rnn(x, hidden)
        return self.classifier(z.view(z.size(0), -1)), hidden

    def init_hidden(self):
        return Variable(torch.randn(1, 1, self.num_hidden))


class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self):
        genepool = GenePool(redis_params=self.config.redis_params)
        environment = gym.make(config.env)

        state_shape = (1, 100, 100)
        num_hidden = self.config.num_hidden
        num_actions = environment.action_space.n
        base_filters = 16
        model_class = dict(mlp=MLP, cnn=CNN)[self.config.model]
        model = model_class(state_shape, base_filters, num_hidden, num_actions)
        agent = Agent(model)
        agent.randomize(config.gene_weight_ratio, config.freq_weight_ratio, config.v_init)
        agent.update_model()
        print(agent.summary())

        if self.config.exhibition:
            self.exhibition(agent, environment, genepool)
        else:
            def run_episode_with_args(agent, env):
                def f(genome):
                    return self.run_episode(agent, genome, env)
                return f
            worker = Optimizer(config, model, agent, genepool)
            worker.run(run_episode_with_args(agent, environment))

    def exhibition(self, agent, environment, genepool):
        while True:
            best_genomes = genepool.top_n(config.num_best)
            best_genome, _ = random.choice(best_genomes)
            agent.load_genome(best_genome)
            agent.update_model()
            print(agent.genome.summary())
            print('Starting episode')
            reward, num_steps = self.run_episode(agent, agent.genome, environment)
            print('Reward {} in {} steps'.format(reward, num_steps))
            genepool.report_score(agent.genome, reward)

    def run_episode(self, agent, genome, environment):
        total_reward = 0.
        num_steps = 0
        observation = environment.reset()
        done = False
        hidden = agent.model.init_hidden()
        player_ready = False
        start_delay = np.random.randint(self.config.random_start)
        agent.genome = genome
        agent.update_model()
        while not done:
            if self.config.render or self.config.exhibition:
                environment.render()
            observation = imresize(observation.astype(np.float32), (100, 100), interp='bicubic')
            observation = np.mean(observation, axis=2, keepdims=True).transpose(2, 0, 1)
            if np.random.uniform() < 0.1:
                imsave('observation.png', observation.transpose(1, 2, 0)[:,:,0].astype(np.uint8))
            observation = observation / 255. - 0.5
            # take action
            player_ready = start_delay <= 0
            if player_ready:
                action, hidden = agent.policy_rnn(observation, hidden)
            else:
                action = np.random.randint(environment.action_space.n)
                start_delay -= 1
            observation, reward, done, info = environment.step(action)
            total_reward += reward
            num_steps += 1
        return total_reward, num_steps


def main(config):
    experiment = Experiment(config)
    experiment.run()


if __name__ == '__main__':
    import argparse
    import multiprocessing

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', default='SpaceInvaders-v0')
    argparser.add_argument('--render', action='store_true')
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--exhibition', action='store_true')
    argparser.add_argument('--random-start', type=int, default=30)
    argparser.add_argument('--model', default='mlp')
    argparser.add_argument('--base-agent-id', type=int, default=0)
    Optimizer.add_config_to_parser(argparser)
    config = argparser.parse_args()

    if config.clear_store:
        genepool = GenePool(redis_params=config.redis_params)
        genepool.clear()

    if config.exhibition:
        main(config)
    else:
        processes = []
        for agent_i in range(config.num_agents):
            p = multiprocessing.Process(target=main, args=(config,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
