import redis


class GenePool:
    def __init__(self, key='genescores', redis_params=None):
        self.key = key
        if redis_params is None:
            redis_params = dict(host='localhost')
        self.redis = redis.StrictRedis(**redis_params)

    def report_score(self, genome, score):
        data = genome.serialize_genomes()
        self.redis.zadd(self.key, score, data)

    def top_n(self, n, reverse=True):
        if reverse:
            f = self.redis.zrevrange
        else:
            f = self.redis.zrange
        results = f(self.key, 0, n - 1, withscores=True)
        return results

    def clear(self):
        self.redis.delete(self.key)
