# distutils: language = c++

from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.set cimport set as cset
from libcpp.vector cimport vector

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, combinations, groupby, islice, product as it_product
import math
from multiprocessing import Manager, Pool
import numpy as np
from queue import Empty
from threading import Thread
from typing import Dict, List, Tuple


PROCESSES = 4


class Progress:
    def __init__(self, manager):
        self.curr = manager.Value("i", 0)
        self.max = manager.Value("i", 0)


cdef class SectorAlias:
    cdef:
        public (int, int) location
        public str type
        public vector[int] content
        public int targets
        public (float, float) center
        public float minimal_radius

    # def __cinit__(self, location, type, content, targets, center, minimal_radius):
    #     self.location = location
    #     self.type = type
    #     self.content = content
    #     self.targets = targets
    #     self.center = center
    #     self.minimal_radius = minimal_radius

    cpdef float distance(self, other) noexcept:
        return math.dist(other.center, self.center)
    
    cpdef bint is_adjacent(self, other) noexcept:
        return math.isclose(self.distance(other), 2 * self.minimal_radius, rel_tol=0.05)
    
    def __eq__(self, other):
        return isinstance(other, SectorAlias) and self.location == other.location
    
    def __hash__(self):
        return hash(self.location)


cdef class Env:
    cdef:
        public list sectors
        public vector[int] colonized_sectors
        public vector[int] empty_sectors
        public float sector_radius
        public int warps_count
        public vector[int] stations
        public vector[float] capacities
        public vector[int] targets
        public vector[bint] storages
        public vector[float] likely_assign

    # def __cinit__(self, colonized_sectors, empty_sectors, sector_radius, warps_count, stations, capacities, targets, storages, likely_assign):
    #     self.colonized_sectors = colonized_sectors
    #     self.empty_sectors = empty_sectors
    #     self.sector_radius = sector_radius
    #     self.warps_count = warps_count
    #     self.stations = stations
    #     self.capacities = capacities
    #     self.targets = targets
    #     self.storages = storages
    #     self.likely_assign = likely_assign


cdef class ConsumerArgs:
    cdef:
        public vector[int] ts_sectors
        public vector[int] warps
    
    # def __cinit__(self, ts_sectors, warps):
    #     self.ts_sectors = ts_sectors
    #     self.warps = warps


cdef class Solution:
    cdef:
        public float score
        public int number
        public list ts_sectors
        public list warps
        public dict assignment_dict


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


def ts_selection(Env env):
    if not env.stations.empty():
        for possibility in combinations(env.empty_sectors, len(env.stations)):
            for sec, ts in zip(possibility, env.stations):
                env.sectors[sec].content = [ts]
                env.sectors[sec].type = "station"
            yield possibility
    else:
        return [[]]

    
def warps_selection(env, sectors, count):
    possibilities = combinations(sectors, count+1)
    groups = groupby(sorted(((c, sum(env.sectors[s].targets for s in c)) for c in possibilities), key=lambda cw: cw[1], reverse=True), key=lambda cw: cw[1])
    for _, g in groups:
        g = list(g)
        for cw in g:
            yield tuple(cw[0])
        return
        
def producer(inqueue, env, total_produced):
    for ts_sectors in ts_selection(env):
        sectors = env.colonized_sectors + list(ts_sectors)
        warps_count = min(len(sectors)-1, env.warps_count)
        for warps in warps_selection(env, sectors, warps_count):
            total_produced.value += 1
            arg = ConsumerArgs()
            arg.ts_sectors = list(ts_sectors)
            arg.warps = list(warps)
            inqueue.put(arg)
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
            neighbourhood = dict(FindNeighbours(env, arg))
            assignments = ProductSlice(*(np.array([j for j, _ in neighbourhood[i] if not env.storages[j]]) for i in sorted(neighbourhood.keys())))
            progress.max.value = assignments.max
            assignments.step = int(max(assignments.max/10000, 1))
            for assignment in assignments:
                progress.curr.value = assignments.i
                solution = evaluate(env, arg, neighbourhood, assignment)
                if solution is not None:
                    solution.number = counter.next()
                    outqueue.put(solution)
        except Empty:
            pass

cdef evaluate(Env env, ConsumerArgs arg, neighbourhood, assignment):
    cdef cmap[int, vector[int]] assignment_dict
    cdef vector[float] capacities_left
    capacities_left = env.capacities
    for i, j in enumerate(assignment):
        capacities_left[j] -= env.targets[i]
    if all(r > 0.3 for r in capacities_left):
        score1 = 1
        for i, r in enumerate(capacities_left):
            if not in_vector(arg.warps, find_sid(env.sectors, i)):
                score1 *= r/env.targets[i] if math.isclose(r, 1, rel_tol=0.05) else 0.5
            else:
                score1 *= r/env.targets[i]
        score2 = 1
        score3 = 1
        score4 = 0
        for i, j in enumerate(assignment):
            sid1 = find_sid(env.sectors, i)
            sec1 = env.sectors[sid1]
            sid2 = find_sid(env.sectors, j)
            sec2 = env.sectors[sid2]
            for k, warped in neighbourhood[i]:
                if k == j:
                    d = max(1, sec1.distance(sec2)/(env.sector_radius*2)) if sid1 != sid2 else 1
                    score2 *= d if warped else 0.1 ** d
            score3 *= env.likely_assign[j]
            score4 += 1 if in_vector(arg.warps, sid1) and in_vector(arg.warps, sid2) else 0
            if assignment_dict.find(j) == assignment_dict.end():
                assignment_dict.insert((j, [i]))
            else:
                assignment_dict[j].push_back(i)
        score = score1 * score2 * score3 * score4
        solution = Solution()
        solution.score = score
        solution.ts_sectors = [env.sectors[sec] for sec in arg.ts_sectors]
        solution.warps = [env.sectors[sec] for sec in arg.warps]
        solution.assignment_dict = assignment_dict
        return solution
    else:
        return None


cdef class FindNeighbours:

    cdef list sectors
    cdef int[:] warps
    cdef cmap[int, cset[int]] neighbour_sectors
    cdef vector[cpair[int, int]] stack

    def __cinit__(self, Env env, ConsumerArgs arg):
        self.sectors = env.sectors
        self.warps = np.array(arg.warps)
        self.stack = [(sid, cid) for sid, s in enumerate(self.sectors) for cid in s.content]
        pass
    
    def __iter__(self):
        return self
    
    def __next__(self):
        cdef cset[int] neighbours
        cdef vector[cpair[int, bint]] sos
        cdef cpair[int, bint] p
        
        if not self.stack.empty():
            sid, cid = self.stack.back()
            self.stack.pop_back()
            if self.neighbour_sectors.find(sid) == self.neighbour_sectors.end():
                neighbours.insert(sid)
                sector = self.sectors[sid]
                for oid, other in enumerate(self.sectors):
                    if sector.is_adjacent(other):
                        neighbours.insert(oid)
                if in_memview(self.warps, sid):
                    for wid in self.warps:
                        neighbours.insert(wid)
                self.neighbour_sectors.insert((sid, neighbours))
            for oid in self.neighbour_sectors[sid]:
                for ocid in self.sectors[oid].content:
                    if cid != ocid:
                        p = (ocid, in_memview(self.warps, sid) and in_memview(self.warps, oid))
                        sos.push_back(p)
            return cid, sos
        else:
            raise StopIteration()
    
    def __getitem__(self, key):
        return self.neighbour_sectors[key]


cdef SectorAlias find_sector(list sectors, int so):
    for sector in sectors:
        if so in sector.content:
            return sector


cdef int find_sid(list sectors, int so):
    for sid, sector in enumerate(sectors):
        if so in sector.content:
            return sid


cdef class ProductSlice:

    cdef list it
    cdef size_t n
    cdef size_t start
    cdef size_t stop
    cdef size_t step
    cdef public size_t i
    cdef public size_t max

    def __cinit__(self, *iterables, size_t start=0, size_t stop=-1, size_t step=1):
        self.it = list(iterables)
        self.n = len(self.it)
        self.max = math.prod(len(l) for l in self.it)
        self.start = start
        self.stop = stop if stop >= 0 else self.max
        self.step = step
        self.i = self.start
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.stop:
            result = np.zeros(self.n, dtype=int)
            j = self.i
            for k, l in enumerate(self.it):
                result[k] = l[j % len(l)]
                j //= len(l)
            if j > 0: raise StopIteration()
            self.i += self.step
            return result
        else:
            raise StopIteration()


cdef bint in_vector(vector[int] v, int n):
    for k in v:
        if k == n:
            return True
    return False


cdef bint in_memview(int[:] v, int n):
    for k in v:
        if k == n:
            return True
    return False
