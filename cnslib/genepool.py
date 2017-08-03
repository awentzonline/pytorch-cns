import hashlib

import numpy as np
import redis
import scipy.stats


class GenePool:
    def __init__(self, key='genescores', redis_params=None):
        self.key = key
        if redis_params is None:
            redis_params = dict(host='localhost')
        self.redis = redis.StrictRedis(**redis_params)

    def report_score(self, genome, score):
        '''Saves a history of each genome's fitness and saves the mean in
        a sorted set'''
        data = genome.serialize_genomes()
        genome_key = self.genome_key(data)
        self.redis.rpush(genome_key, score)
        history = self.redis.lrange(genome_key, 0, -1)
        history = list(map(float, history))
        self.redis.zadd(self.key, np.mean(history), data)

    def top_n(self, n, reverse=True):
        if reverse:
            f = self.redis.zrevrange
        else:
            f = self.redis.zrange
        results = f(self.key, 0, n - 1, withscores=True)
        return results

    def clear(self):
        self.redis.flushdb()

    def genome_key(self, serialized_genome):
        return '{}/gene/{}'.format(self.key, hashlib.sha1(serialized_genome).hexdigest())
