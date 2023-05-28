# distutils: language = c++

from libcpp.algorithm cimport lower_bound
from libc.math cimport fabs as cabs
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.set cimport set as cset
from libcpp.vector cimport vector

from collections import defaultdict
import cython
from cython.operator import dereference
from cython.parallel import parallel
from itertools import combinations, groupby
import math
from multiprocessing import Queue
from queue import Empty
from threading import Thread

cdef extern from "<atomic>" namespace "std" nogil:
    cdef cppclass atomic_long:
        atomic_long() nogil noexcept
        atomic_long(long) nogil noexcept
        long load() nogil noexcept
        void store(long) nogil noexcept
        long fetch_add(long) nogil noexcept

ctypedef atomic_long* atomic_long_ptr

cdef extern from "<chrono>" namespace "std::chrono" nogil:
    cdef cppclass milliseconds:
        milliseconds(int) nogil noexcept

cdef extern from "<thread>" namespace "std::this_thread" nogil:
    cdef void sleep_for(milliseconds&) nogil noexcept


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint float_greater_than(float a, float b) nogil noexcept:
    return a > b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint insort_float(vector[float]& vec, float value) nogil noexcept:
    cdef bint result
    it = lower_bound(vec.begin(), vec.end(), value, float_greater_than)
    result = it - vec.begin() < 10
    vec.insert(it, value)
    if vec.size() > 10:
        vec.pop_back()
    return result


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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float distance(vector[cpair[float, float]] centers, Py_ssize_t id1, Py_ssize_t id2) nogil noexcept:
    return ((centers[id1].first - centers[id2].first)**2 + (centers[id1].second - centers[id2].second)**2)**(0.5)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_adjacent(vector[cpair[float, float]] center, Py_ssize_t id1, Py_ssize_t id2, float radius) nogil noexcept:
    cdef float a, b
    a = distance(center, id1, id2)
    b = 2.0 * radius
    return cabs(a - b) <= 0.05 * max(cabs(a), cabs(b))


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
    cdef public list ts_sectors
    cdef public list warps
    cdef public dict directions
    
def make_solution(score, ts_sectors, warps, assignment):
    solution = Solution()
    solution.score = score
    solution.ts_sectors = ts_sectors
    solution.warps = warps
    directions = defaultdict(list)
    for i, j in enumerate(assignment):
        directions[j].append(i)
    solution.directions = {k: v for k, v in directions.items()}
    return solution

ctypedef cpair[atomic_long, atomic_long]* al_al_pair_ptr

cdef class Compute:
    cdef public object task
    cdef public object solutions
    cdef public list sectors
    cdef public list targets
    cdef object produced
    cdef Env env
    cdef atomic_long total_produced
    cdef vector[al_al_pair_ptr] consumer_progresses
    cdef vector[atomic_long_ptr] total_consumed

    def __cinit__(self, app, int processes=4):
        self.produced = Queue(100000)
        self.solutions = Queue(100000)
        self.total_produced.store(0)

        self.make_env(app)

        def run_consumers():
            cdef atomic_long_ptr tprogress
            cdef cpair[atomic_long, atomic_long]* cprogress
            env = self.env
            produced = self.produced
            solutions = self.solutions
            with cython.nogil, parallel():
                cprogress = new cpair[atomic_long, atomic_long]()
                tprogress = new atomic_long(0)
                with gil:
                    self.consumer_progresses.push_back(cprogress)
                    self.total_consumed.push_back(tprogress)
                consume(env, produced, solutions, dereference(cprogress), dereference(tprogress))

        Thread(target=self._produce, daemon=True).start()
        self.task = Thread(target=run_consumers)
        self.task.daemon = True
        self.task.start()

    def progress(self):
        produced = self.total_produced.load()
        consumed = sum(c.load() for c in self.total_consumed)
        return consumed/produced if produced > 0 else 0
    
    def subprogresses(self) -> vector[cpair[int, int]]:
        return [(dereference(p).first.load(), dereference(p).second.load()) for p in self.consumer_progresses]

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
                arg = ConsumerArgs(ts_sectors, warps)
                self.produced.put(arg)
                self.total_produced.fetch_add(1)
        self.produced.put(StopIteration())


    def ts_selection(self):
        if not self.env.targets.stations.empty():
            for possibility in combinations(self.env.sectors.empty, len(self.env.targets.stations)):
                yield tuple(possibility)
        else:
            return [[]]

    
    def warps_selection(self, ts_sectors, count):
        possibilities = combinations(concat_vectors(self.env.sectors.colonized, ts_sectors), count+1 if count else 0)
        groups = groupby(sorted(((c, sum(self.env.sectors.targets_count[s] if self.env.sectors.targets_count[s] else 1 for s in c)) for c in possibilities), key=lambda cw: cw[1], reverse=True), key=lambda cw: cw[1])
        for _, g in groups:
            g = list(g)
            for cw in g:
                yield tuple(cw[0])
            return


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void consume(Env& env, object produced, object solutions, cpair[atomic_long, atomic_long]& cprogress, atomic_long& tprogress) nogil noexcept:
    cdef Neighbourhood neighbourhood
    cdef ConsumerArgs arg
    cdef cpair[bint, vector[Py_ssize_t]] assignment
    cdef float score
    cdef vector[float] top_scores

    top_scores.push_back(0.0)

    while True:
        with gil:
            try:
                fromq = produced.get(False)
            except Empty:
                with nogil:
                    sleep_for(milliseconds(1))
                    continue
            if isinstance(fromq, StopIteration):
                produced.put(fromq)
                break
            arg = fromq
        neighbourhood = find_neighbours(env, arg)
        assignments = make_assignments(env, neighbourhood)
        cprogress.second.store(assignments.max)
        cprogress.first.store(0)
        assignments.step = int(max(assignments.max/10000, 1))
        assignment = next_product(assignments)
        while assignment.first:
            score = evaluate(env, arg, neighbourhood, assignment.second)
            if insort_float(top_scores, score):
                with cython.gil:
                    solution = make_solution(score, list(arg.ts_sectors), list(arg.warps), assignment.second)
                    solutions.put(solution)
            assignment = next_product(assignments)
            cprogress.first.fetch_add(assignments.step)
        tprogress.fetch_add(1)
        

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float evaluate(Env& env, ConsumerArgs& arg, cmap[Py_ssize_t, vector[cpair[Py_ssize_t, bint]]]& neighbourhood, vector[Py_ssize_t]& assignment) nogil noexcept:
    cdef float score, score1, score2, score3, score4
    cdef Py_ssize_t i, j, k, sid1, sid2
    cdef bint warped, valid
    cdef cpair[Py_ssize_t, Py_ssize_t] size_size_pair
    cdef cpair[Py_ssize_t, float] size_float_pair
    cdef cpair[Py_ssize_t, bint] size_bint_pair
    cdef vector[float] capacities_left
    
    capacities_left = env.targets.capacities
    assignment_enumerated = enumerate_sizes(assignment)
    for size_size_pair in assignment_enumerated:
        capacities_left[size_size_pair.second] -= env.targets.targets[size_size_pair.first]
    with gil:
        if any(r < 0.3 for r in capacities_left):
            return 0
    score1 = 1
    capacities_left_enumerated = enumerate_floats(capacities_left)
    for size_float_pair in capacities_left_enumerated:
        if not in_vector(arg.warps, find_id(env.sectors.content, size_float_pair.first)):
            with gil:
                score1 *= size_float_pair.second/env.targets.targets[size_float_pair.first] if math.isclose(size_float_pair.second, 1, rel_tol=0.05) else 0.5
        else:
            score1 *= size_float_pair.second/env.targets.targets[size_float_pair.first]
    score2 = 1
    score3 = 1
    score4 = 1
    assignment_enumerated = enumerate_sizes(assignment)
    for size_size_pair in assignment_enumerated:
        sid1 = find_id(env.sectors.content, size_size_pair.first)
        sid2 = find_id(env.sectors.content, size_size_pair.second)
        for size_bint_pair in neighbourhood[size_size_pair.first]:
            k = size_bint_pair.first
            warped = size_bint_pair.second
            if size_bint_pair.first == size_size_pair.second:
                d = max(1.0, distance(env.sectors.center, sid1, sid2)/(env.sectors.radius*2)) if sid1 != sid2 else 1.0
                score2 *= d if warped else 0.1 ** d
        score3 *= env.targets.likely_assign[size_size_pair.second]
        score4 += 1 if in_vector(arg.warps, sid1) and in_vector(arg.warps, sid2) else 0
    score = score1 * score2 * score3 * score4
    return score


ctypedef cmap[Py_ssize_t, vector[cpair[Py_ssize_t, bint]]] Neighbourhood


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Product make_assignments(Env& env, Neighbourhood& neighbourhood) nogil noexcept:
    cdef vector[vector[Py_ssize_t]] vs
    cdef vector[Py_ssize_t] v
    cdef cpair[Py_ssize_t, bint] p

    vs.reserve(env.targets.storages.size())
    for i in range(env.targets.storages.size()):
        v.clear()
        v.reserve(neighbourhood[i].size())
        for p in neighbourhood[i]:
            if not env.targets.storages[p.first]:
                v.push_back(p.first)
        vs.push_back(v)
    return product(vs)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Neighbourhood find_neighbours(Env& env, ConsumerArgs& arg) nogil noexcept:
    cdef Neighbourhood result
    
    cdef cmap[Py_ssize_t, vector[Py_ssize_t]] content
    cdef vector[Py_ssize_t] sectors
    cdef cset[Py_ssize_t] neighbours
    cdef cpair[Py_ssize_t, bint] size_bint_pair
    cdef cpair[Py_ssize_t, vector[Py_ssize_t]] size_vector_pair
    
    for sector in env.sectors.colonized:
        size_vector_pair.first = sector
        size_vector_pair.second = env.sectors.content[sector]
        content.insert(size_vector_pair)
    ts_sectors_enumerated = enumerate_sizes(arg.ts_sectors)
    for size_size_pair in ts_sectors_enumerated:
        size_vector_pair.first = size_size_pair.second
        size_vector_pair.second.clear()
        size_vector_pair.second.push_back(env.targets.stations[size_size_pair.first])
        content.insert(size_vector_pair)
    sectors = concat_vectors(env.sectors.colonized, arg.ts_sectors)
    for sid in sectors:
        neighbours.clear()
        neighbours.insert(sid)
        for oid in sectors:
            if is_adjacent(env.sectors.center, sid, oid, env.sectors.radius):
                neighbours.insert(oid)
        if in_vector(arg.warps, sid):
            for wid in arg.warps:
                neighbours.insert(wid)
        for cid in content[sid]:
            for oid in neighbours:
                for ocid in content[oid]:
                    if cid != ocid:
                        size_bint_pair.first = ocid
                        size_bint_pair.second = in_vector(arg.warps, sid) and in_vector(arg.warps, oid)
                        result[cid].push_back(size_bint_pair)
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t find_id(vector[vector[Py_ssize_t]]& contents, Py_ssize_t n) nogil noexcept:
    cdef Py_ssize_t id = 0
    for content in contents:
        if in_vector(content, n):
            return id
        id += 1


cdef struct Product:
    vector[vector[Py_ssize_t]] it
    Py_ssize_t n
    Py_ssize_t step
    Py_ssize_t curr
    Py_ssize_t max


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Product product(vector[vector[Py_ssize_t]]& it, Py_ssize_t step=1) nogil noexcept:
    cdef Py_ssize_t m = 1
    cdef Product prod
    for v in it:
        m *= v.size()
    prod.it = it
    prod.n = it.size()
    prod.step = step
    prod.curr = 0
    prod.max = m
    return prod


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cpair[bint, vector[Py_ssize_t]] next_product(Product& prod) nogil noexcept:
    cdef cpair[bint, vector[Py_ssize_t]] result
    cdef vector[Py_ssize_t] l
    cdef Py_ssize_t j
    
    if prod.curr < prod.max:
        result.second.reserve(prod.n)
        j = prod.curr
        for l in prod.it:
            result.second.push_back(l[j % l.size()])
            j //= l.size()
        result.first = not (j > 0)
        prod.curr += prod.step
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint in_vector(vector[Py_ssize_t]& v, Py_ssize_t n) nogil noexcept:
    for k in v:
        if k == n:
            return True
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline vector[Py_ssize_t] concat_vectors(vector[Py_ssize_t]& a, vector[Py_ssize_t]& b) nogil noexcept:
    cdef vector[Py_ssize_t] result
    result.reserve(a.size() + b.size())
    for i in a:
        result.push_back(i)
    for i in b:
        result.push_back(i)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline vector[cpair[Py_ssize_t, Py_ssize_t]] enumerate_sizes(vector[Py_ssize_t]& v) nogil noexcept:
    cdef vector[cpair[Py_ssize_t, Py_ssize_t]] result
    cdef cpair[Py_ssize_t, Py_ssize_t] pair
    cdef Py_ssize_t i = 0
    result.reserve(v.size())
    for k in v:
        pair.first = i
        pair.second = k
        result.push_back(pair)
        i += 1
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline vector[cpair[Py_ssize_t, float]] enumerate_floats(vector[float]& v) nogil noexcept:
    cdef vector[cpair[Py_ssize_t, float]] result
    cdef cpair[Py_ssize_t, float] pair
    cdef Py_ssize_t i = 0
    cdef float k
    result.reserve(v.size())
    for k in v:
        pair.first = i
        pair.second = k
        result.push_back(pair)
        i += 1
    return result