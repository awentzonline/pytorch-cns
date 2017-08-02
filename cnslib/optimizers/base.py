import json

import numpy as np

from ..argtypes import list_of
from ..genome import ModelGenome


class BaseOptimizer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.rng = np.random.RandomState(self.config.seed)

    def run(self, f_run_episode):
        num_episodes = 0
        self.pre_run()
        while True:
            print('Episode {}'.format(num_episodes))
            self.pre_episode(num_episodes)
            reward, steps = f_run_episode(self.genome)
            self.post_episode(num_episodes, reward, steps)
            print('Reward {} in {} steps'.format(reward, steps))
            num_episodes += 1

    def pre_run(self):
        pass

    def pre_episode(self, episode):
        pass

    def post_episode(self, episode, reward, steps):
        pass

    @classmethod
    def add_config_to_parser(cls, parser):
        parser.add_argument('--gene-weight-ratio', type=float, default=0.001)
        parser.add_argument('--freq-weight-ratio', type=float, default=1.)
        parser.add_argument('--i-sigma', type=float, default=1.)
        parser.add_argument('--v-sigma', type=list_of(float), default=1.)
        parser.add_argument('--v-init', type=list_of(float), default=(-1, 1.))
        parser.add_argument('--seed', type=int, default=int('beef', 16))
        parser.add_argument('--clear-store', action='store_true')
        parser.add_argument('--redis-params', type=json.loads, default={'host': 'localhost'})
