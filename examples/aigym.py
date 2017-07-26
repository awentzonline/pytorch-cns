import random
import time

import gym
import numpy as np
import redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from cnslib.genome import ModelGenome


class Genepool:
    def __init__(self, key='genescores', redis_params=None):
        self.key = key
        if redis_params is None:
            redis_params = dict(host='localhost')
        self.redis = redis.StrictRedis(**redis_params)

    def report_score(self, genome, score):
        data = genome.serialize_genomes()
        self.redis.zadd(self.key, score, data)

    def top_n(self, n):
        results = self.redis.zrevrange(self.key, 0, n - 1, withscores=True)
        return results

    def clear(self):
        self.redis.delete(self.key)


class Agent:
    def __init__(self, model):
        self.model = model
        self.genome = ModelGenome(model)
        self.genome_a = ModelGenome(model)
        self.genome_b = ModelGenome(model)

    def policy(self, state):
        state = state[None, ...].astype(np.float32)
        state = torch.from_numpy(state)
        inputv = Variable(state)
        ps = self.model(inputv).data.numpy()
        ps = np.argmax(ps)
        return ps

    def randomize(self, config):
        self.genome.randomize(5, 20, (-1., 1.))

    def crossover(self, best_genomes):
        parents = random.sample(best_genomes, 2)
        best_genomes = [genome for genome, _ in best_genomes]
        self.genome_a.deserialize_genomes(best_genomes[0])
        self.genome_b.deserialize_genomes(best_genomes[1])
        self.genome.child(self.genome_a, self.genome_b)

    def mutate(self):
        self.genome.mutate(value_range=(-1., 1.))

    def update_model(self):
        self.genome.decode(self.model)


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
    num_hidden = 32
    num_actions = environment.action_space.n
    model = MLP(state_shape[0], num_hidden, num_actions)
    agent = Agent(model)
    agent.randomize(config)
    genepool = Genepool()
    if config.clear_store:
        genepool.clear()
    while True:
        print('Starting episode')
        reward, steps = run_episode(agent, environment, config)
        print('Reward {} in {} steps'.format(reward, steps))
        best_genomes = genepool.top_n(config.min_genepool)
        if len(best_genomes) < config.min_genepool:
            genepool.report_score(agent.genome, steps)  # we're still gathering scores
        else:
            _, best_score = best_genomes[0]
            _, worst_best_score = best_genomes[-1]
            print('Genepool top: {}, {}'.format(best_score, worst_best_score))
            if reward < worst_best_score or np.random.uniform() < 0.1:
                # Our score isn't notable
                agent.crossover(best_genomes)
            else:
                # New high-ish score
                print('new ok score')
                genepool.report_score(agent.genome, reward)
        agent.mutate()
        agent.update_model()
        time.sleep(np.random.uniform())


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
    argparser.add_argument('--min-genepool', type=int, default=10)
    argparser.add_argument('--render', action='store_true')
    argparser.add_argument('--clear-store', action='store_true')
    config = argparser.parse_args()
    main(config)
