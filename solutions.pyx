# distutils: language = c++

from libcpp.algorithm cimport lower_bound
from libcpp.list cimport list as clist
from libc.math cimport fabs as cabs, hypot
from libcpp.map cimport map as cmap
from libcpp.pair cimport pair as cpair
from libcpp.deque cimport deque
from libcpp.set cimport set as cset
from libcpp.vector cimport vector
from libc.stdio cimport printf

from collections import defaultdict
import cython
from cython.operator import dereference
from cython.parallel import parallel
from itertools import combinations, groupby
import math
from multiprocessing import Queue
from queue import Full, Empty
from threading import Thread

cdef extern from "<algorithm>" namespace "std" nogil:
    cdef bint next_permutation[T](T, T) nogil noexcept

cdef extern from "<atomic>" namespace "std" nogil:
    cdef cppclass atomic[T]:
        atomic() nogil noexcept
        atomic(T) nogil noexcept
        T load() nogil noexcept
        void store(T) nogil noexcept
        T fetch_add(T) nogil noexcept

ctypedef atomic[Py_ssize_t] atomic_size
ctypedef atomic_size* atomic_size_ptr

cdef extern from "<chrono>" namespace "std::chrono" nogil:
    cdef cppclass milliseconds:
        milliseconds(int) nogil noexcept

cdef extern from "<thread>" namespace "std::this_thread" nogil:
    cdef void sleep_for(milliseconds&) nogil noexcept

cdef extern from "<mutex>" namespace "std" nogil:
    cdef cppclass timed_mutex:
        timed_mutex() nogil noexcept
        void lock() nogil noexcept
        bint try_lock() nogil noexcept
        bint try_lock_for(milliseconds&) nogil noexcept
        void unlock() nogil noexcept


cdef extern from "<iterator>" namespace "std" nogil:
    ctypedef input_iterator
    cdef void advance[T](T&, int) nogil noexcept


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint double_greater_than(double a, double b) nogil noexcept:
    return a > b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint insort_double(vector[double]& vec, double value) nogil noexcept:
    cdef bint result
    it = lower_bound(vec.begin(), vec.end(), value, double_greater_than)
    result = it - vec.begin() < 10
    vec.insert(it, value)
    if vec.size() > 10:
        vec.pop_back()
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[bint] select(Py_ssize_t n, Py_ssize_t k) nogil noexcept:
    cdef vector[bint] result
    cdef Py_ssize_t i = k
    result.assign(n-k, False)
    while i < n:
        result.push_back(True)
        i += 1
    return result


ctypedef vector[Py_ssize_t] sizes_t
ctypedef cmap[Py_ssize_t, sizes_t] sizes_mapping_t
ctypedef cpair[Py_ssize_t, sizes_t] sizes_mapping_element_t


cdef enum SectorType:
    star, hidden, empty, planets, station

cdef struct EnvSectors:
    double radius
    sizes_t empty
    sizes_t colonized
    vector[sizes_t] content
    vector[SectorType] type
    vector[int] targets_count
    vector[cpair[double, double]] center


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double distance(vector[cpair[double, double]]& centers, Py_ssize_t id1, Py_ssize_t id2) nogil noexcept:
    return hypot(centers[id1].first - centers[id2].first, centers[id1].second - centers[id2].second)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_close(double a, double b, double rel_tol=0.05) nogil noexcept:
    return cabs(a - b) <= rel_tol * max(cabs(a), cabs(b))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_adjacent(vector[cpair[double, double]]& center, Py_ssize_t id1, Py_ssize_t id2, double radius) nogil noexcept:
    return is_close(distance(center, id1, id2), 2.0 * radius)


cdef struct EnvTargets:
    sizes_t stations
    vector[double] capacities
    vector[int] targets
    vector[bint] storages
    vector[double] likely_assign

cdef struct Env:
    Py_ssize_t warps_count
    EnvSectors sectors
    EnvTargets targets


ctypedef vector[double] distances_t


cdef struct ConsumerArgs:
    sizes_t ts_sectors
    sizes_t warps
    distances_t offlane_distances
    bint guard


cdef struct SolutionDataObject:
    double score
    sizes_t ts_sectors
    sizes_t warps
    sizes_t assignment
    bint guard


cdef struct sdo_sync_queue:
    timed_mutex* m
    deque[SolutionDataObject] q
    bint c

    # def __cinit__(self):
    #     self.c = False
    #     # printf("sdo_sync_queue.__cinit__")

cdef inline void sdosq_mark_exhaused(sdo_sync_queue& sdosq) nogil noexcept:
    sdosq.m.lock()
    sdosq.c = True
    sdosq.m.unlock()
    
cdef inline void sdosq_push(sdo_sync_queue& sdosq, SolutionDataObject& sdo) nogil noexcept:
    while not sdosq.m.try_lock_for(milliseconds(100)):
        sleep_for(milliseconds(1000))
    sdosq.q.push_back(sdo)
    sdosq.m.unlock()
    
cdef inline void sdosq_push_many(sdo_sync_queue& sdosq, vector[SolutionDataObject]& sdos) nogil noexcept:
    while not sdosq.m.try_lock_for(milliseconds(100)):
        sleep_for(milliseconds(1000))
    for sdo in sdos:
        sdosq.q.push_back(sdo)
    sdosq.m.unlock()
    
cdef bint sdosq_take(sdo_sync_queue& sdosq, n, vector[SolutionDataObject]& result):
    result.clear()
    result.reserve(n)
    if sdosq.m.try_lock_for(milliseconds(100)):
        if sdosq.q.empty() and sdosq.c:
            sdosq.m.unlock()
            return False
        i = 0
        while i < n and not sdosq.q.empty():
            result.push_back(sdosq.q.front())
            sdosq.q.pop_front()
            i += 1
        sdosq.m.unlock()
    return True

class Solution:
    score: float
    ts_sectors: list
    warps: list
    directions: dict


cdef object make_solution(SolutionDataObject& sdo):
    solution = Solution()
    solution.score = sdo.score
    solution.ts_sectors = list(sdo.ts_sectors)
    solution.warps = list(sdo.warps)
    dd = defaultdict(list)
    for i, j in enumerate(sdo.assignment):
        dd[j].append(i)
    solution.directions = dict(dd)
    return solution


cdef struct Arrow:
    Py_ssize_t start
    Py_ssize_t end
    double length


ctypedef cpair[atomic_size, atomic_size]* asize_pair_ptr

ctypedef deque[SolutionDataObject]* collector_qt
ctypedef timed_mutex* collector_lt
ctypedef cpair[collector_qt, collector_lt] collector_pt

cdef class Compute:
    cdef public int threads_count
    cdef public object task
    cdef public list sectors
    cdef public list targets
    cdef sdo_sync_queue solutions
    cdef deque[ConsumerArgs] produced
    cdef timed_mutex lock_produced
    cdef Env env
    cdef int total_produced
    cdef vector[asize_pair_ptr] consumer_progresses
    cdef vector[atomic_size_ptr] total_consumed
    cdef clist[collector_pt] collectors

    def __cinit__(self, app):
        self.total_produced = 0
        self.solutions.m = new timed_mutex()
        self.make_env(app)

        Thread(target=self._produce, daemon=True).start()
        Thread(target=self._run_consumers, daemon=True).start()
        self.task = Thread(target=self._collect, daemon=True)
        self.task.start()

    def progress(self):
        consumed = sum(c.load() for c in self.total_consumed)
        return consumed/self.total_produced if self.total_produced > 0 else 0
    
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

    def take_solutions(self, n=10):
        cdef vector[SolutionDataObject] buffer
        if sdosq_take(self.solutions, n, buffer):
            for sdo in buffer:
                yield make_solution(sdo)
        else:
            raise Empty


    def _produce(self):
        warps_count = min(len(self.env.sectors.colonized)+len(self.env.targets.stations)-1, self.env.warps_count) + 1 if self.env.warps_count else 0
        self.total_produced = 1
        self.total_produced *= math.comb(self.env.sectors.empty.size(), self.env.targets.stations.size()) if not self.env.targets.stations.empty() else 1
        n = 0
        for s in self.env.sectors.colonized:
            if self.env.sectors.targets_count[s] > 1:
                n += 1
        self.total_produced *= math.comb(self.env.sectors.colonized.size() - min(n, warps_count) + self.env.targets.stations.size(), warps_count)
        
        buffer = []
        for ts_sectors in self.ts_selection():
            for warps in self.warps_selection(ts_sectors, warps_count):
                sectors = list(self.env.sectors.colonized) + list(ts_sectors)
                distances = [0.0] * len(self.env.sectors.type)
                if len(warps):
                    for sid1 in sectors:
                        if sid1 not in warps:
                            distances[sid1] = min(distance(self.env.sectors.center, sid1, sid2) / (self.env.sectors.radius * 2) for sid2 in warps)
                buffer.append(ConsumerArgs(ts_sectors, warps, distances, False))
            self.lock_produced.lock()
            for buffered in buffer:
                self.produced.push_back(buffered)
            size = self.produced.size()
            self.lock_produced.unlock()
            buffer.clear()
            if size > 10000:
                with nogil:
                    sleep_for(milliseconds(10000))
        self.lock_produced.lock()
        self.produced.push_back(ConsumerArgs([], [], [], True))
        self.lock_produced.unlock()


    def ts_selection(self):
        if not self.env.targets.stations.empty():
            for possibility in combinations(self.env.sectors.empty, len(self.env.targets.stations)):
                yield tuple(possibility)
        else:
            return [[]]

    
    def warps_selection(self, ts_sectors, count):
        cdef sizes_t sectors
        concat_vectors(self.env.sectors.colonized, ts_sectors, sectors)
        possibilities = combinations(sectors, count)
        groups = groupby(sorted(((c, sum(self.env.sectors.targets_count[s] if self.env.sectors.targets_count[s] else 1 for s in c)) for c in possibilities), key=lambda cw: cw[1], reverse=True), key=lambda cw: cw[1])
        for _, g in groups:
            g = list(g)
            for cw in g:
                yield tuple(cw[0])
            return

    def _run_consumers(self):
        cdef atomic_size_ptr tprogress
        cdef asize_pair_ptr cprogress
        cdef collector_pt collector
        env = self.env
        produced = &self.produced
        lock_produced = &self.lock_produced
        with cython.nogil, parallel():
            cprogress = new cpair[atomic_size, atomic_size]()
            tprogress = new atomic_size(0)
            collectq = new deque[SolutionDataObject]()
            collectl = new timed_mutex()
            collector = collector_pt(collectq, collectl)
            with gil:
                self.consumer_progresses.push_back(cprogress)
                self.total_consumed.push_back(tprogress)
                self.collectors.push_back(collector)
            consume(env, produced, lock_produced, collector, dereference(cprogress), dereference(tprogress))

    def _collect(self):
        while self.threads_count == 0:
            pass
        with nogil:
            collect(self.collectors, self.solutions)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void collect(clist[collector_pt]& collectors, sdo_sync_queue& solutions) nogil noexcept:
    cdef collector_pt collector
    cdef vector[SolutionDataObject] buffer
    cdef vector[double] top_scores

    buffer.reserve(10000)
    top_scores.push_back(0.0)

    while True:
        it = collectors.begin()
        while it != collectors.end():
            collector = dereference(it)
            collector.second.lock()
            for _ in range(min(collector.first.size(), 1000)):
                sdo = collector.first.front()
                if sdo.guard:
                    it = collectors.erase(it)
                    break
                collector.first.pop_front()
                if insort_double(top_scores, sdo.score):
                    buffer.push_back(sdo)
            collector.second.unlock()
            advance(it, 1)
        if not buffer.empty():
            sdosq_push_many(solutions, buffer)
            buffer.clear()
        if collectors.size() == 0:
            break
        if solutions.m.try_lock_for(milliseconds(10000)):
            if solutions.q.size() > 100:
                sleep_for(milliseconds(1000))
    if not buffer.empty():
        sdosq_push_many(solutions, buffer)
        buffer.clear()
    sdosq_mark_exhaused(solutions)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void consume(Env& env, deque[ConsumerArgs]* produced, timed_mutex* lock_produced, collector_pt collector, cpair[atomic_size, atomic_size]& cprogress, atomic_size& tprogress) nogil noexcept:
    cdef Neighbourhood neighbourhood
    cdef ConsumerArgs arg
    cdef double score
    cdef vector[double] top_scores
    cdef SolutionDataObject sdo
    cdef vector[SolutionDataObject] buffer

    top_scores.push_back(0.0)

    while True:
        lock_produced.lock()
        while produced.empty():
            lock_produced.unlock()
            sleep_for(milliseconds(1000))
            lock_produced.lock()
        arg = produced.front()
        if not arg.guard:
            produced.pop_front()
        lock_produced.unlock()
        if arg.guard:
            break
        neighbourhood = find_neighbours(env, arg)
        assignments = make_assignments(env, neighbourhood)
        buffer.reserve(min(assignments.max, 1000000))
        cprogress.second.store(assignments.max)
        cprogress.first.store(0)
        assignments.step = <Py_ssize_t>(max(assignments.max/100000, 1.0))
        while next_product(assignments):
            score = evaluate(env, arg, neighbourhood, assignments.result, (top_scores.back() if top_scores.size() == 10 else -1))
            if insort_double(top_scores, score):
                sdo.score = score
                sdo.ts_sectors = arg.ts_sectors
                sdo.warps = arg.warps
                sdo.assignment = assignments.result
                sdo.guard = False
                buffer.push_back(sdo)
            cprogress.first.fetch_add(assignments.step)
        collector.second.lock()
        for buffered in buffer:
            collector.first.push_back(buffered)
        buffer.clear()
        collector.second.unlock()
        tprogress.fetch_add(1)
    collector.second.lock()
    for buffered in buffer:
        collector.first.push_back(buffered)
    buffer.clear()
    sdo.guard = True
    collector.first.push_back(sdo)
    collector.second.unlock()
        

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double evaluate(Env& env, ConsumerArgs& arg, cmap[Py_ssize_t, vector[cpair[Py_ssize_t, bint]]]& neighbourhood, sizes_t& assignment, double worst) nogil noexcept:
    cdef double score, score1, score2, score3, score4, score5, r, d, offlane_length
    cdef Py_ssize_t i, j, k, sid1, sid2
    cdef bint warped, valid
    cdef cpair[Py_ssize_t, Py_ssize_t] size_size_pair
    cdef cpair[Py_ssize_t, double] size_double_pair
    cdef cpair[Py_ssize_t, bint] size_bint_pair
    cdef vector[double] capacities_left
    cdef sizes_t sectors
    cdef vector[Arrow] offlane_arrows
    cdef sizes_t arrows_order
    
    capacities_left = env.targets.capacities
    assignment_enumerated = enumerate_sizes(assignment)
    for size_size_pair in assignment_enumerated:
        capacities_left[size_size_pair.second] -= env.targets.targets[size_size_pair.first]
    for r in capacities_left:
        if r < 0.3:
            return 0
    score1 = 1
    capacities_left_enumerated = enumerate_doubles(capacities_left)
    for size_double_pair in capacities_left_enumerated:
        if in_vector(arg.warps, find_id(env.sectors.content, size_double_pair.first)):
            score1 *= size_double_pair.second/env.targets.targets[size_double_pair.first] if is_close(size_double_pair.second, 1, rel_tol=0.05) else 0.5
        else:
            score1 *= size_double_pair.second/env.targets.targets[size_double_pair.first]
    score2 = 1
    score3 = 1
    score4 = 1
    score5 = 1
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
        score4 *= 1 if in_vector(arg.warps, sid1) and in_vector(arg.warps, sid2) else 0.5
    score = score1 * score2 * score3 * score4
    if score < worst:
        return 0
    concat_vectors(env.sectors.colonized, arg.ts_sectors, sectors)
    for sid1 in sectors:
        if not in_vector(arg.warps, sid1):
            score5 *= 1/arg.offlane_distances[sid1] if arg.offlane_distances[sid1] > 0 else 1
    score *= score5
    if score < worst:
        return 0
    find_offlane_arrows(env, arg, assignment, offlane_arrows)
    if offlane_arrows.size() > 1:
        arrows_order = crange(offlane_arrows.size())
        offlane_length = -1
        while True:
            d = compute_offlane_route(env, arg, offlane_arrows, arrows_order)
            offlane_length = d if offlane_length == -1 else min(offlane_length, d)
            if not next_permutation(arrows_order.begin(), arrows_order.end()):
                break
        score *= (1/offlane_length if offlane_length > 0 else 1)
    return score


ctypedef cmap[Py_ssize_t, vector[cpair[Py_ssize_t, bint]]] Neighbourhood


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Product make_assignments(Env& env, Neighbourhood& neighbourhood) nogil noexcept:
    cdef vector[sizes_t] vs
    cdef sizes_t v
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
cdef inline Neighbourhood find_neighbours(Env& env, ConsumerArgs& arg) nogil noexcept:
    cdef Neighbourhood result
    
    cdef cmap[Py_ssize_t, sizes_t] content
    cdef sizes_t sectors
    cdef cset[Py_ssize_t] neighbours
    cdef cpair[Py_ssize_t, bint] size_bint_pair
    cdef cpair[Py_ssize_t, sizes_t] size_vector_pair
    
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
    concat_vectors(env.sectors.colonized, arg.ts_sectors, sectors)
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
cdef inline vector[Arrow] find_offlane_arrows(Env& env, ConsumerArgs& arg, sizes_t& assignment, vector[Arrow]& result) nogil noexcept:
    cdef Arrow arrow
    cdef vector[cpair[Py_ssize_t, Py_ssize_t]] enumerated
    cdef cpair[Py_ssize_t, Py_ssize_t] pair
    cdef Py_ssize_t sid1, sid2
    
    result.clear()
    enumerated = enumerate_sizes(assignment)
    for pair in enumerated:
        sid1 = find_id(env.sectors.content, pair.first)
        sid2 = find_id(env.sectors.content, pair.second)
        if not in_vector(arg.warps, sid1) or not in_vector(arg.warps, sid2):
            arrow.start = pair.second
            arrow.end = pair.first
            arrow.length = distance(env.sectors.center, pair.second, pair.first)/env.sectors.radius*2
            result.push_back(arrow)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double compute_offlane_route(Env& env, ConsumerArgs& arg, vector[Arrow]& offlane_arrows, sizes_t& arrows_order) nogil noexcept:
    cdef double dl, wl, length = 0
    cdef Py_ssize_t i, n, prev

    i = 0
    n = arrows_order.size()
    prev = offlane_arrows[arrows_order[0]].start

    while i < n:
        dl = distance(env.sectors.center, prev, offlane_arrows[arrows_order[i]].start)/env.sectors.radius*2
        wl = arg.offlane_distances[prev] + 1 + arg.offlane_distances[offlane_arrows[arrows_order[i]].start]
        length += min(dl, wl) + offlane_arrows[arrows_order[i]].length
        prev = offlane_arrows[arrows_order[i]].end
        i += 1
    
    dl = distance(env.sectors.center, prev, offlane_arrows[arrows_order[0]].start)/env.sectors.radius*2
    wl = arg.offlane_distances[prev] + 1 + arg.offlane_distances[offlane_arrows[arrows_order[0]].start]
    length += min(dl, wl)

    return length


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t find_id(vector[sizes_t]& contents, Py_ssize_t n) nogil noexcept:
    cdef Py_ssize_t id = 0
    for content in contents:
        if in_vector(content, n):
            return id
        id += 1


cdef struct Product:
    vector[sizes_t] it
    sizes_t result
    Py_ssize_t n
    Py_ssize_t step
    Py_ssize_t curr
    Py_ssize_t max


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Product product(vector[sizes_t]& it, Py_ssize_t step=1) nogil noexcept:
    cdef Py_ssize_t m = 1
    cdef Product prod
    for v in it:
        m *= v.size()
    prod.it = it
    prod.n = it.size()
    prod.step = step
    prod.curr = 0
    prod.max = m
    prod.result.assign(prod.n, 0)
    return prod


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint next_product(Product& prod) nogil noexcept:
    cdef bint result
    cdef sizes_t l
    cdef Py_ssize_t j, i
    
    result = False
    if prod.curr < prod.max:
        j = prod.curr
        i = 0
        for l in prod.it:
            prod.result[i] = l[j % l.size()]
            j //= l.size()
            i += 1
        result = not (j > 0)
        prod.curr += prod.step
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint in_vector(sizes_t& v, Py_ssize_t n) nogil noexcept:
    for k in v:
        if k == n:
            return True
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void concat_vectors(sizes_t& a, sizes_t& b, sizes_t& c) nogil noexcept:
    c.clear()
    c.reserve(a.size() + b.size())
    for i in a:
        c.push_back(i)
    for i in b:
        c.push_back(i)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline vector[cpair[Py_ssize_t, Py_ssize_t]] enumerate_sizes(sizes_t& v) nogil noexcept:
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
cdef inline vector[cpair[Py_ssize_t, double]] enumerate_doubles(vector[double]& v) nogil noexcept:
    cdef vector[cpair[Py_ssize_t, double]] result
    cdef cpair[Py_ssize_t, double] pair
    cdef Py_ssize_t i = 0
    cdef double k
    result.reserve(v.size())
    for k in v:
        pair.first = i
        pair.second = k
        result.push_back(pair)
        i += 1
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline sizes_t crange(Py_ssize_t n) nogil noexcept:
    cdef sizes_t result
    cdef Py_ssize_t i = 0
    result.reserve(n)
    while i < n:
        result.push_back(i)
        i += 1
    return result
