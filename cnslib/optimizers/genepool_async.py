import json

import numpy as np

from ..argtypes import list_of
from ..genome import ModelGenome
from ..scoreboard import AgentScoreboard
from .base import BaseOptimizer


class Optimizer(BaseOptimizer):
    '''"Asynchronous Gene Pool"

    This optimizer evaluates genome fitness in parallel and asynchronously.
    The gene pool--a set of all genomes in the popuation ordered by their
    fitness--is maintained by a group of workers which pull highly-fit genomes
    and report the fitness of their mutant offspring.
    '''
    def __init__(self, config, model, agent, genepool):
        super(Optimizer, self).__init__(config, model)
        self.agent = agent
        self.genepool = genepool

    def post_episode(self, episode, reward, num_steps):
        self.genepool.report_score(self.agent.genome, reward)
        best_genomes = self.genepool.top_n(self.config.num_best)
        _, best_score = best_genomes[0]
        _, worst_best_score = best_genomes[-1]
        print('Genepool top: {}, {}'.format(best_score, worst_best_score))
        if best_genomes and self.rng.uniform() < 0.1:
            # replay top genomes to make sure they're not flukes
            best_genome, _ = best_genomes[self.rng.randint(len(best_genomes))]
            self.agent.load_genome(best_genome)
        else:
            if reward < 0.5 * (worst_best_score + best_score) and len(best_genomes) > 1:
                self.agent.crossover(best_genomes)
            self.agent.mutate(index_sigma=self.config.i_sigma, value_sigma=self.config.v_sigma)
        self.agent.update_model()

    @property
    def genome(self):
        return self.agent.genome

    @classmethod
    def add_config_to_parser(cls, parser):
        super(Optimizer, cls).add_config_to_parser(parser)
        parser.add_argument('--num-best', type=int, default=30)
        parser.add_argument('--num-agents', type=int, default=10)
