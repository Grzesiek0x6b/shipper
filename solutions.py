from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations, groupby, islice, product
import math
from multiprocessing import Manager, Pool
from queue import Empty
from threading import Thread
from typing import Dict, List, Tuple


PROCESSES = 4


class Progress:
    def __init__(self, manager):
        self.curr = manager.Value("i", 0)
        self.max = manager.Value("i", 0)


@dataclass(order=True)
class SectorAlias:
    location: Tuple[int, int] = (0, 0)
    type: str = "hidden"
    content: List[int] = field(default_factory=list)
    targets: int = 0
    center: Tuple[float, float] = (0, 0)
    minimal_radius: float = 1

    def distance(self, other):
        return math.dist(other.center, self.center)
    
    def is_adjacent(self, other):
        return math.isclose(self.distance(other), 2 * self.minimal_radius, rel_tol=0.05)
    
    def __eq__(self, other):
        return isinstance(other, SectorAlias) and self.location == other.location
    
    def __hash__(self):
        return hash(self.location)


@dataclass
class Env:
    colonized_sectors: List[SectorAlias]
    empty_sectors: List[SectorAlias]
    sector_radius: float
    warps_count: int
    stations: List[int]
    capacities: List[float]
    targets: List[int]
    storages: List[bool]
    likely_assign: List[float]

    @property
    def sectors(self):
        return self.colonized_sectors + self.empty_sectors


@dataclass
class ConsumerArgs:
    ts_sectors: List[SectorAlias]
    warps: List[SectorAlias]


@dataclass
class Solution:
    score: float
    number: int
    ts_sectors: List[SectorAlias]
    warps: List[SectorAlias]
    assignment_dict: Dict[int, List[int]]


class Counter:
    def __init__(self, n=1, s=0):
        self.i = 1
        self.n = n
        self.s = s
    
    def next(self):
        val = self.i * self.n + self.s
        self.i += 1
        return val


def compute(env):
    m = Manager()
    inqueue = m.Queue(100000)
    outqueue = m.Queue(100000)
    consumers_progress = [Progress(m) for _ in range(PROCESSES)]
    total_produced = m.Value("i", 0)
    total_consumed = [m.Value("i", 0) for _ in range(PROCESSES)]

    def run_consumers():
        with Pool(processes=PROCESSES) as pool:
            pool.map_async(consumer, ((inqueue, outqueue, env, Counter(PROCESSES, i), individual_progress, total_consumed[i]) for i, individual_progress in enumerate(consumers_progress))).get()
        pool.close()
        pool.join()

    pthread = Thread(target=producer, args=(inqueue, env, total_produced))
    pthread.daemon = True
    pthread.start()

    cthread = Thread(target=run_consumers)
    cthread.daemon = True
    cthread.start()

    def progress():
        produced = total_produced.value
        consumed = sum(c.value for c in total_consumed)
        return consumed/produced if produced > 0 else 0

    return cthread, outqueue, consumers_progress, progress


def ts_selection(env):
    if env.stations:
        for possibility in combinations(env.empty_sectors, len(env.stations)):
            for sec, ts in zip(possibility, env.stations):
                sec.content = [ts]
                sec.type = "station"
            yield possibility
    else:
        return [[]]

    
def warps_selection(sectors, count):
    possibilities = combinations(sectors, count+1)
    groups = groupby(sorted(((c, sum(s.targets for s in c)) for c in possibilities), key=lambda cw: cw[1], reverse=True), key=lambda cw: cw[1])
    for _, g in groups:
        g = list(g)
        for cw in g:
            yield tuple(cw[0])
        return
        
def producer(inqueue, env, total_produced):
    for ts_sectors in ts_selection(env):
        sectors = env.colonized_sectors + list(ts_sectors)
        warps_count = min(len(sectors)-1, env.warps_count)
        for warps in warps_selection(sectors, warps_count):
            total_produced.value += 1
            inqueue.put(ConsumerArgs(ts_sectors, warps))
    inqueue.put(StopIteration())


def consumer(args):
    inqueue, outqueue, env, counter, progress, consumed_progress = args
    while True:
        try:
            arg = inqueue.get(True, 1)
            if isinstance(arg, StopIteration):
                inqueue.put(arg)
                break
            consumed_progress.value += 1
            neighbourhood = dict(find_neighbours(env.sectors, arg.warps))
            for assignment in sample_assigments(env, neighbourhood, progress):
                capacities_left = env.capacities.copy()
                for i, j in enumerate(assignment):
                    capacities_left[j] -= env.targets[i]
                if all(r > 0.3 for r in capacities_left):
                    score1 = 1
                    for i, r in enumerate(capacities_left):
                        if find_sector(env.sectors, i) not in arg.warps:
                            score1 *= r/env.targets[i] if math.isclose(r, 1, rel_tol=0.05) else 0.5
                        else:
                            score1 *= r/env.targets[i]
                    score2 = 1
                    score3 = 1
                    score4 = 0
                    assignment_dict = defaultdict(list)
                    for i, j in enumerate(assignment):
                        sec1 = find_sector(env.sectors, i)
                        sec2 = find_sector(env.sectors, j)
                        for k, warped in neighbourhood[i]:
                            if k == j:
                                d = max(1, sec1.distance(sec2)/(env.sector_radius*2)) if sec1 != sec2 else 1
                                score2 *= d if warped else 0.1 ** d
                        score3 *= env.likely_assign[j]
                        score4 += 1 if sec1 in arg.warps and sec2 in arg.warps else 0
                        assignment_dict[j].append(i)
                    score = score1 * score2 * score3 * score4
                    outqueue.put(Solution(score, counter.next(), arg.ts_sectors, arg.warps, assignment_dict))
        except Empty:
            pass


def find_neighbours(sectors, warps):
    for sector in sectors:
        neighbours = set(other for other in sectors if sector.is_adjacent(other))
        neighbours.add(sector)
        if sector in warps:
            neighbours.update(warps)
        for i in sector.content:
            yield i, tuple((j, sector in warps and othersec in warps) for othersec in neighbours for j in othersec.content if i!=j)


def find_sector(sectors, so):
    for sector in sectors:
        if so in sector.content:
            return sector


def sample_assigments(env, neighbourhood, progress):
    assignments = product(*((j for j, _ in neighbourhood[i] if not env.storages[j]) for i in neighbourhood.keys()))
    max_progress = math.prod(sum(1 for j, _ in neighbourhood[i] if not env.storages[j]) for i in neighbourhood.keys())
    step = int(max(max_progress/10000, 1))
    progress.max.value = max_progress
    progress.curr.value = 0
    for assigment in islice(assignments, None, None, step):
        progress.curr.value += step
        yield assigment

