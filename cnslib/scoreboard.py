import json
import time

import redis


class AgentScoreboard:
    def __init__(self, num_agents, key='agent_scores', redis_params=None):
        self.num_agents = num_agents
        self.key = key
        if redis_params is None:
            redis_params = dict(host='localhost')
        self.redis = redis.StrictRedis(**redis_params)

    def report_score(self, agent_id, generation, score):
        self.redis.hset(self.key, agent_id, score)
        self.redis.hincrby(self.generation_complete_key, generation, 1)

    def is_generation_complete(self, generation):
        num_complete = self.redis.hget(self.generation_complete_key, generation)
        return num_complete and int(num_complete) >= self.num_agents

    def wait_for_generation(self, generation, poll_delay=0.5):
        # TODO: do this without polling?
        while True:
            if self.is_generation_complete(generation):
                break
            time.sleep(poll_delay)

    def fetch_scores(self):
        raw = self.redis.hgetall(self.key)
        return dict((int(k), float(v)) for k, v in raw.items())

    def fetch_ordered_scores(self, reverse=True):
        raw = self.redis.hgetall(self.key)
        return sorted(((float(v), int(k)) for k, v in raw.items()), reverse=reverse)

    def top_n(self, n, reverse=True):
        '''Defaults to assuming high scores are better.'''
        scores = self.fetch_scores()
        scores = sorted(
            [(score, agent_id) for agent_id, score in scores.items()], reverse=reverse
        )
        return scores[:n]

    def clear(self):
        self.redis.delete(self.key)
        self.redis.delete(self.generation_complete_key)
        self.redis.delete(self.top_genome_key)

    def save_top_genomes(self, genomes):
        data = [g.serialize_genomes().decode() for g in genomes]
        self.redis.set(self.top_genome_key, json.dumps(data))

    def report_top_genome(self, genome, score):
        data = genome.serialize_genomes().decode()
        self.redis.zadd(self.top_genome_key, json.dumps(data), score)

    def fetch_top_genomes(self):
        data = self.redis.get(self.top_genome_key).decode()
        return json.loads(data)

    @property
    def generation_complete_key(self):
        return '{}/complete'.format(self.key)

    @property
    def top_genome_key(self):
        return '{}/top'.format(self.key)


if __name__ == '__main__':
    import random

    num_agents = 10
    generation = 0
    asb = AgentScoreboard(  # db=9 so as not to collide with default db=0
        num_agents, key='test_scoreboard', redis_params=dict(db=9, host='localhost')
    )
    asb.clear()

    asb.report_score(0, generation, 10.5)
    scores = asb.fetch_scores()
    assert scores[generation] and scores[generation] == 10.5
    asb.clear()

    assert not asb.is_generation_complete(generation)
    scores = []
    for i in range(num_agents):
        assert not asb.is_generation_complete(generation)
        score = random.random() * 30.0 - 15.0
        asb.report_score(i, generation, score)
        scores.append((i, score))
    assert asb.is_generation_complete(generation)
    remote_scores = asb.fetch_scores()
    for agent_id, score in scores:
        assert remote_scores[agent_id] == score
    top_score, top_agent = asb.top_n(1)[0]
    assert top_score == max(score for _, score in scores)
    assert len(asb.top_n(5)) == 5
    asb.clear()
    assert asb.redis.dbsize() == 0
