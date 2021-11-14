"""
This code is implementation of Parallel Radius Stepping with (1,rho) preprocessing

Author : Yang Yuyun
Date Created : 11-02-2021

Base RadiusStepping Algorithm Acknowledgements:
Guy E. Blelloch - guyb@cs.cmu.edu

@inproceedings{blelloch2016parallel,
  title={Parallel shortest paths using radius stepping},
  author={Blelloch, Guy E and Gu, Yan and Sun, Yihan and Tangwongsan, Kanat},
  booktitle={Proceedings of the 28th ACM Symposium on Parallelism in Algorithms and Architectures},
  pages={443--454},
  year={2016}
}
"""

from math import floor, sqrt
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Array
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from pprint import pprint
from bst import *
from sortedcontainers import SortedDict
import time
class RStepping:
    """

    """

    def __init__(self):
        """

        """
        self.distances = {}
        self.delta = 5
        self.property_map = {}
        self.workItems = []
        self.source_vertex = 3
        self.infinity = float("inf")
        self.totalNodes = 0
        self.totalEdges = 0
        self.Q = dict()
        self.R = dict()
        self.r = mp.Manager().dict()

    def preprocess(self, g):
        """
        1,rho
        """

        def find_lowest_cost_node(costs, processed):
            lowest_cost = self.infinity
            lowest_cost_node = None
            for node in costs:
                if not (node in processed):
                    if costs[node] < lowest_cost:
                        lowest_cost = costs[node]
                        lowest_cost_node = node
            return lowest_cost_node

        
        def dijkstra(s, rho, r):
            processed = []
            costs = {}
            costs[s] = 0
            r_tmp = []
            for u in g.neighbors(s):
                costs[u] = g.get_edge_data(s, u)['weight']
            node = find_lowest_cost_node(costs, processed)
            while node is not None and len(processed) <= rho:
               cost = costs[node]
                for n in g.neighbors(node):
                    new_cost = cost + g.get_edge_data(node, n)['weight']
                    if new_cost < costs.get(n, self.infinity):
                        costs[n] = new_cost
                        processed.append(node)
                if node != s and node not in g.neighbors(s):
                    g.add_edge(s, node, weight=costs[node])
                r_tmp.append(costs[node])
                node = find_lowest_cost_node(costs, processed)

            r[s] = max(r_tmp)
            pass


        pool = Pool(processes=4)
        for i in g.nodes:
            pool.apply_async(dijkstra, (i, 20, self.r))
        pool.close()
        pool.join()


    def radius_init(self, g):
        """
        This is the main function to implement the algorithm

        :param g:
        :return:
        """
        for node in g.nodes():
            self.property_map[node] = self.infinity

        self.property_map[self.source_vertex] = 0
        for u in g.neighbors(self.source_vertex):
            w = g.get_edge_data(self.source_vertex, u)['weight']
            self.Q[u] = w
            self.R[u] = w + self.r[u]
            self.property_map[u] = w

    def radius_stepping(self, g):

        def neighbor(a,property_map,temp,di,R,Q,v,r):
            if property_map[v] > temp:

                if property_map[v] > di and temp <= di:
                    if v in R:
                        del R[v]
                    if v in Q:
                        del Q[v]
                    a.append(v)
                property_map[v] = temp
                if property_map[v] > di:
                    Q[v] = property_map[v]
                    R[v] = property_map[v] + r[v]
            return

        while len(self.Q) > 0:
            
            di = min(self.R.values())
            a = []
            for key in self.Q.keys():
                if self.Q[key] <= di:
                    a.append(key)
            
            for key in a:
                del self.Q[key]
                del self.R[key]

            pool = Pool(processes=4)
            while len(a) > 0:#

                node = a.pop() 

                for v in g.neighbors(node):
                    temp = self.property_map[node] + g.get_edge_data(node, v)['weight']
                    neighbor(a,self.property_map, temp, di, self.R, self.Q,v,self.r)
            pool.close()
            pool.join()

        return

def main():

    g = nx.read_edgelist('artist.txt', nodetype=int, data=(('weight', int),))


    print(nx.info(g))

    start = time.time()
    a = RStepping()
    a.preprocess(g)
    end = time.time()
    print(end - start)
    start = time.time()

    a.radius_init(g)
    a.radius_stepping(g)
    end = time.time()
    print(end - start)

    #visualize the graph

if __name__ == '__main__':
    main()