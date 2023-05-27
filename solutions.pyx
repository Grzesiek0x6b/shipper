# distutils: language = c++

from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.set cimport set as cset
from libcpp.vector cimport vector

cimport cython
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, combinations, groupby, islice, product as it_product
import math
from multiprocessing import Manager, Pool, Queue, Value
import numpy as np
from queue import Empty
from threading import Thread
from typing import Dict, List, Tuple


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

    cpdef float distance(self, other) noexcept:
        return math.dist(other.center, self.center)
    
    cpdef bint is_adjacent(self, other) noexcept:
        return math.isclose(self.distance(other), 2 * self.minimal_radius, rel_tol=0.05)
    
    def __eq__(self, other):
        return isinstance(other, SectorAlias) and self.location == other.location
    
    def __hash__(self):
        return hash(self.location)


cdef enum SectorType:
    star, hidden, empty, planets, station

cdef struct EnvSectors:
    float radius
    vector[Py_ssize_t] empty
    vector[Py_ssize_t] colonized
    vector[vector[Py_ssize_t]] content
    vector[SectorType] type
    vector[int] targets_count
    vector[cpair[float, float]] center


cdef inline float distance(vector[cpair[float, float]] center, Py_ssize_t id1, Py_ssize_t id2):
    return math.dist((center[id1].first, center[id1].second), (center[id2].first,center[id2].second))


cdef inline bint is_adjacent(vector[cpair[float, float]] center, Py_ssize_t id1, Py_ssize_t id2, float radius):
    return math.isclose(distance(center, id1, id2), 2 * radius, rel_tol=0.05)


cdef struct EnvTargets:
    vector[Py_ssize_t] stations
    vector[float] capacities
    vector[int] targets
    vector[bint] storages
    vector[float] likely_assign

cdef struct Env:
    Py_ssize_t warps_count
    EnvSectors sectors
    EnvTargets targets


cdef struct ConsumerArgs:
    vector[Py_ssize_t] ts_sectors
    vector[Py_ssize_t] warps


cdef class Solution:
    cdef public float score
    cdef public vector[Py_ssize_t] ts_sectors
    cdef public vector[Py_ssize_t] warps
    cdef public cmap[Py_ssize_t, vector[Py_ssize_t]] assignment_dict

cdef class Compute:
    cdef public object task
    cdef public object solutions
    cdef public list sectors
    cdef public list targets
    cdef object produced
    cdef Env env
    cdef object total_produced
    cdef list total_consumed
    cdef list consumer_progresses

    def __cinit__(self, app, processes=4):
        m = Manager()
        self.produced = m.Queue(100000)
        self.solutions = m.Queue(100000)
        self.consumer_progresses = [Progress(m) for _ in range(processes)]
        self.total_produced = m.Value("i", 0)
        self.total_consumed = [m.Value("i", 0) for _ in range(processes)]

        self.make_env(app)

        def run_consumers():
            threads = []
            for i, progress in enumerate(self.consumer_progresses):
                thread = Thread(target=self._consume, args=(progress, self.total_consumed[i]))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            
            
            # with Pool(processes=PROCESSES) as pool:
            #     pool.map_async(self._consume, ((progress, self.total_consumed[i]) for i, progress in enumerate(self.consumer_progresses))).get()
            # pool.close()
            # pool.join()

        Thread(target=self._produce, daemon=True).start()
        self.task = Thread(target=run_consumers)
        self.task.daemon = True
        self.task.start()

    def progress(self) -> float:
        produced = self.total_produced.value
        consumed = sum(c.value for c in self.total_consumed)
        return consumed/produced if produced > 0 else 0
    
    def subprogresses(self) -> vector[cpair[int, int]]:
        return [(p.curr.value, p.max.value) for p in self.consumer_progresses]

    def make_env(self, app):
        self.env.warps_count = app.warps_count

        empty_sectors = [sector for sector in app.sectors if sector.sector_type == "empty"]
        colonized_sectors = [sector for sector in app.sectors if sector.sector_type=="planets" and any(planet.colonized for planet in sector.sector_content)]
        adjacent_empty_sectors = set(empty for empty in empty_sectors for colonized in colonized_sectors if empty.is_adjacent(colonized))
        self.sectors = list(set(colonized_sectors).union(adjacent_empty_sectors))
        
        self.env.sectors.radius = app.sectors[0].minimal_radius
        self.env.sectors.empty = [self.sectors.index(sec) for sec in adjacent_empty_sectors]
        self.env.sectors.colonized = [self.sectors.index(sec) for sec in colonized_sectors]
        types = {"star": SectorType.star, "hidden": SectorType.hidden, "empty": SectorType.empty, "planets": SectorType.planets, "station": SectorType.station}
        self.env.sectors.type = [types[sector.sector_type] for sector in self.sectors]
        self.env.sectors.targets_count = [sum(obj.targets for obj in sector.sector_content) for sector in self.sectors]
        self.env.sectors.center = [sector.center for sector in self.sectors]

        colonized_planets = [planet for sector in colonized_sectors for planet in sector.sector_content if planet.colonized]
        self.targets = colonized_planets + app.stations
        all_packages = sum(obj.produces for obj in self.targets)
        per_target = all_packages / sum(obj.targets for obj in self.targets)

        self.env.sectors.content = [[self.targets.index(obj) for obj in sector.sector_content] for sector in self.sectors]

        self.env.targets.stations = [self.targets.index(station) for station in app.stations]
        self.env.targets.capacities = [obj.capacity/per_target for obj in self.targets]
        self.env.targets.targets = [obj.targets for obj in self.targets]
        self.env.targets.storages = [obj.storage for obj in self.targets]
        self.env.targets.likely_assign = [obj.likely_assign for obj in self.targets]

    def _produce(self):
        for ts_sectors in self.ts_selection():
            warps_count = min(len(self.env.sectors.colonized)+len(ts_sectors)-1, self.env.warps_count)
            for warps in self.warps_selection(ts_sectors, warps_count):
                self.total_produced.value += 1
                arg = ConsumerArgs(ts_sectors, warps)
                # arg.ts_sectors = ts_sectors
                # arg.warps = warps
                self.produced.put(arg)
        self.produced.put(StopIteration())


    def _consume(self, progress, total_consumed):
        # progress, total_consumed = args
        while True:
            try:
                arg = self.produced.get(True, 1)
                if isinstance(arg, StopIteration):
                    self.produced.put(arg)
                    break
                total_consumed.value += 1
                neighbourhood = dict(FindNeighbours(self.env, arg))
                assignments = ProductSlice(*(np.array([j for j, _ in neighbourhood[i] if not self.env.targets.storages[j]]) for i in sorted(neighbourhood.keys())))
                progress.max.value = assignments.max
                assignments.step = int(max(assignments.max/10000, 1))
                for assignment in assignments:
                    progress.curr.value = assignments.i
                    solution = self.evaluate(arg, neighbourhood, assignment)
                    if solution is not None:
                        self.solutions.put(solution)
            except Empty:
                pass


    def ts_selection(self):
        if not self.env.targets.stations.empty():
            for possibility in combinations(self.env.sectors.empty, len(self.env.targets.stations)):
                # for sec, ts in zip(possibility, self.env.targets.stations):
                #     self.env.sectors.content[sec] = [ts]
                #     self.env.sectors.types[sec] = "station"
                
                yield tuple(possibility)
        else:
            return [[]]

    
    def warps_selection(self, ts_sectors, count):
        possibilities = combinations(list(chain(self.env.sectors.colonized, ts_sectors)), count+1 if count else 0)
        groups = groupby(sorted(((c, sum(self.env.sectors.targets_count[s] if self.env.sectors.targets_count[s] else 1 for s in c)) for c in possibilities), key=lambda cw: cw[1], reverse=True), key=lambda cw: cw[1])
        for _, g in groups:
            g = list(g)
            for cw in g:
                yield tuple(cw[0])
            return
        

    cdef evaluate(self, ConsumerArgs arg, neighbourhood, assignment):
        cdef cmap[Py_ssize_t, vector[Py_ssize_t]] assignment_dict
        cdef vector[float] capacities_left
        capacities_left = self.env.targets.capacities
        for i, j in enumerate(assignment):
            capacities_left[j] -= self.env.targets.targets[i]
        if all(r > 0.3 for r in capacities_left):
            score1 = 1
            for i, r in enumerate(capacities_left):
                if not in_vector(arg.warps, find_id(self.env.sectors.content, i)):
                    score1 *= r/self.env.targets.targets[i] if math.isclose(r, 1, rel_tol=0.05) else 0.5
                else:
                    score1 *= r/self.env.targets.targets[i]
            score2 = 1
            score3 = 1
            score4 = 0
            for i, j in enumerate(assignment):
                sid1 = find_id(self.env.sectors.content, i)
                # sec1 = env.sectors[sid1]
                sid2 = find_id(self.env.sectors.content, j)
                # sec2 = env.sectors[sid2]
                for k, warped in neighbourhood[i]:
                    if k == j:
                        d = max(1.0, distance(self.env.sectors.center, sid1, sid2)/(self.env.sectors.radius*2)) if sid1 != sid2 else 1.0
                        score2 *= d if warped else 0.1 ** d
                score3 *= self.env.targets.likely_assign[j]
                score4 += 1 if in_vector(arg.warps, sid1) and in_vector(arg.warps, sid2) else 0
                if assignment_dict.find(j) == assignment_dict.end():
                    assignment_dict.insert((j, [i]))
                else:
                    assignment_dict[j].push_back(i)
            score = score1 * score2 * score3 * score4
            solution = Solution()
            solution.score = score
            solution.ts_sectors = list(arg.ts_sectors)
            solution.warps = list(arg.warps)
            solution.assignment_dict = dict(assignment_dict)
            return solution
        else:
            return None


cdef class FindNeighbours:

    cdef list sectors
    cdef vector[Py_ssize_t] warps
    cdef cmap[Py_ssize_t, vector[Py_ssize_t]] content
    cdef cmap[Py_ssize_t, cset[Py_ssize_t]] neighbour_sectors
    cdef vector[cpair[Py_ssize_t, Py_ssize_t]] stack
    cdef Env env

    def __cinit__(self, Env env, ConsumerArgs arg):
        self.env = env
        self.sectors = list(chain(env.sectors.colonized, arg.ts_sectors))
        self.warps = arg.warps
        for sector in env.sectors.colonized:
            self.content.insert((sector, env.sectors.content[sector]))
        for sector, ts in zip(arg.ts_sectors, env.targets.stations):
            self.content.insert((sector, [ts]))
        self.stack = [(sid, cid) for sid in self.sectors for cid in self.content[sid]]

    
    def __iter__(self):
        return self
    
    def __next__(self):
        cdef cset[Py_ssize_t] neighbours
        cdef vector[cpair[Py_ssize_t, bint]] sos
        cdef cpair[Py_ssize_t, bint] p
        
        if not self.stack.empty():
            sid, cid = self.stack.back()
            self.stack.pop_back()
            if self.neighbour_sectors.find(sid) == self.neighbour_sectors.end():
                neighbours.insert(sid)
                for oid in self.sectors:
                    if is_adjacent(self.env.sectors.center, sid, oid, self.env.sectors.radius):
                        neighbours.insert(oid)
                if in_vector(self.warps, sid):
                    for wid in self.warps:
                        neighbours.insert(wid)
                self.neighbour_sectors.insert((sid, neighbours))
            for oid in self.neighbour_sectors[sid]:
                for ocid in self.content[oid]:
                    if cid != ocid:
                        p = (ocid, in_vector(self.warps, sid) and in_vector(self.warps, oid))
                        sos.push_back(p)
            return cid, sos
        else:
            raise StopIteration()
    
    def __getitem__(self, key):
        return self.neighbour_sectors[key]


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline SectorAlias find_sector(list sectors, Py_ssize_t so) noexcept:
#     for sector in sectors:
#         if so in sector.content:
#             return sector


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int find_id(vector[vector[Py_ssize_t]] contents, Py_ssize_t n) noexcept:
    for id, content in enumerate(contents):
        if in_vector(content, n):
            return id


cdef class ProductSlice:

    cdef list it
    cdef Py_ssize_t n
    cdef Py_ssize_t step
    cdef public Py_ssize_t i
    cdef public Py_ssize_t max

    def __cinit__(self, *iterables, Py_ssize_t step=1):
        self.it = list(iterables)
        self.max = math.prod(len(l) for l in self.it)
        self.step = step
        self.i = 0
        self.n = len(self.it)
    
    def __iter__(self):
        return self

    @cython.wraparound(False)
    def __next__(self):
        if self.i < self.max:
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint in_vector(vector[Py_ssize_t] v, Py_ssize_t n) noexcept:
    for k in v:
        if k == n:
            return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint in_memview(Py_ssize_t[:] v, Py_ssize_t n) noexcept:
    for k in v:
        if k == n:
            return True
    return False
