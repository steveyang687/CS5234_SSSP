"""
This code is implementation of sequential Delta Stepping

Author : Wu YuJiao
Date Created : 10-30-2021

Base DeltaStepping Algorithm Acknowledgements:
Marcin J Zalewski - marcin.zalewski@pnnl.gov

Paper :
@article{meyer_-stepping:_2003,
       title = {Δ-stepping: a parallelizable shortest path algorithm},
       volume = {49},
       issn = {0196-6774},
       shorttitle = {Δ-stepping},
       url = {http://www.sciencedirect.com/science/article/pii/S0196677403000762},
       doi = {10.1016/S0196-6774(03)00076-2},
"""

from math import floor, sqrt
import networkx as nx
import time
from collections import defaultdict


class Algorithm:
    """

    """

    def __init__(self):
        """

        """
        self.delta = 5
        self.property_map = {}
        self.source_vertex = 8
        self.infinity = float("inf")
        self.B = defaultdict(list)

    def delta_stepping(self, g):
        """
        This is the main function to implement the algorithm

        :param g:
        :return:
        """
        for node in g.nodes():
            self.property_map[node] = self.infinity
        self.relax(self.source_vertex, 0)

        while self.B:
            i,r = min(self.B.keys()),[]
            while i in self.B:
                req = self.find_requests(self.B[i], 'light', g)
                r = r + self.B[i]
                del self.B[i]
                self.relax_requests(req)
            req = self.find_requests(r, 'heavy', g)
            self.relax_requests(req)

    def find_requests(self, vertices, kind, g):
        tmp = {}
        for u in vertices:
            for v in g.neighbors(u):

                edge_weight = self.property_map[u] + g.get_edge_data(u, v)['weight']
                if kind == 'light':
                    if g.get_edge_data(u, v)['weight'] <= self.delta:
                        tmp[v] = edge_weight if (v in tmp and edge_weight < tmp[v]) or v not in tmp else tmp[v]

                elif kind == 'heavy':
                    if g.get_edge_data(u, v)['weight'] > self.delta:
                        tmp[v] = edge_weight if (v in tmp and edge_weight < tmp[v]) or v not in tmp else tmp[v]
                else:
                    return None
        # print("tmp=", tmp)
        return tmp

    def relax_requests(self, request):
        for key, value in request.items():
            self.relax(key, value)

    def relax(self, w, x):
        if self.property_map[w] != self.infinity:
            pre = floor(self.property_map[w] / self.delta)
        now = floor(x / self.delta)
        if x < self.property_map[w]:
            if self.property_map[w] != self.infinity:
                if pre in self.B:
                    self.B[pre].remove(w)
            self.B[now].append(w)
            self.property_map[w] = x

def main():
    start = time.time()

    g = nx.read_edgelist('Notre.txt', nodetype=int, data=(('weight', int),))

    print("\nGraph Information..")
    print("===================\n")
    print(nx.info(g))
    print("\nCalculating shortest path..")
    a = Algorithm()
    a.source_vertex = 2
    a.delta_stepping(g)
    end = time.time()
    print(end-start)


if __name__ == '__main__':
    main()
