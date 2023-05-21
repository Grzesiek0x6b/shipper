class Graph:
    def __init__(self,n):
        self.vertices = n
        self.graph = []
    def add_edge(self,from_vertice,to,weight):
        self.graph.append([from_vertice,to,weight])
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent,parent[i])
    def union(self, parent, rank, a, b):
        a_root = self.find(parent, a)
        b_root = self.find(parent, b)
        if rank[a_root] < rank[b_root]:
            parent[a_root] = b_root
        elif rank[a_root] > rank[b_root]:
            parent[b_root] = a_root
        else: 
            parent[b_root] = a_root
            rank[a_root] += 1
    def KruskalAlgo(self):
        MST = []
        e = 0
        r = 0
        self.graph = sorted(self.graph, key = lambda item : item[2]) 
        
        parent = []
        rank = []
        for node in range(self.vertices):
            parent.append(node)
            rank.append(0)
        while r < self.vertices - 1:
            from_vertice = self.graph[e][0]
            to = self.graph[e][1]
            weight = self.graph[e][2]
            e += 1
            a = self.find(parent, from_vertice)
            b = self.find(parent, to)
            if a != b:
                r += 1
                MST.append([from_vertice,to,weight])
                self.union(parent, rank, a, b)
        for from_vertice,to,weight in MST:
            yield (from_vertice, to)