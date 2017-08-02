import json

import numpy as np

from ..argtypes import list_of
from ..genome import ModelGenome
from ..scoreboard import AgentScoreboard
from .base import BaseOptimizer


class Optimizer(BaseOptimizer):
    '''"Synchronous Score Swapper"

    In this optimizer, each process maintains the full state of all genomes
    in the system. The genome fitness is evaluated in parallel and the genetic
    updates occur synchronously at the end of each episode after fitness scores
    are disseminated. This minimizes the amount of data communicated between
    processes.
    '''
    def __init__(self, config, model, agent_id, scoreboard):
        super(Optimizer, self).__init__(config, model)
        self.agent_id = agent_id
        self.scoreboard = scoreboard
        # create genomes for all workers
        self.genomes = []
        self.rng = np.random.RandomState(self.config.seed)
        for agent_id in range(self.config.num_agents):
            genome = ModelGenome(model, rng=self.rng)
            genome.randomize(
                self.config.gene_weight_ratio, self.config.freq_weight_ratio, self.config.v_init
            )
            self.genomes.append(genome)

    def post_episode(self, episode, reward, num_steps):
        self.scoreboard.report_score(self.agent_id, episode, reward)
        self.scoreboard.wait_for_generation(episode)
        self.update_all_genomes()

    def update_all_genomes(self):
        scores = self.scoreboard.fetch_ordered_scores()
        best_scores = scores[:self.config.num_best]
        best_genome_ids = [aid for _, aid in best_scores]
        best_genomes = [self.genomes[aid] for aid in best_genome_ids]
        survivor_ids = set(best_genome_ids[:self.config.num_survivors])
        for agent_id, genome in enumerate(self.genomes):
            if agent_id in survivor_ids:
                self.scoreboard.save_top_genomes(best_genomes[:5])  # fix this
                continue
            a_i, b_i = self.rng.randint(0, len(best_genomes), 2)
            genome_a, genome_b = best_genomes[a_i], best_genomes[b_i]
            genome.child(genome_a, genome_b)
            genome.mutate(index_sigma=self.config.i_sigma, value_sigma=self.config.v_sigma)

    @property
    def genome(self):
        self.genomes[self.agent_id]

    @classmethod
    def add_config_to_parser(cls, parser):
        super(Optimizer, cls).add_config_to_parser(parser)
        parser.add_argument('--num-best', type=int, default=6)
        parser.add_argument('--num-survivors', type=int, default=2)
        parser.add_argument('--num-agents', type=int, default=10)
