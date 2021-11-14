"""
This code is implementation of Parallel Radius Stepping with (k,rho) preprocessing

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
        k,rho
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

        def greedy(s, k, rho, r):
            processed = []
            costs = {s:0}
            hop = {s:0}
            v_to_add = set()  # contain the k*i + 1 vertices to link
            r_tmp = []
            for u in g.neighbors(s):
                costs[u] = g.get_edge_data(s, u)['weight']
                hop[u] = 1
            node = find_lowest_cost_node(costs, processed)

            while node is not None and len(processed) <= rho:
                cost = costs[node]
                for n in g.neighbors(node):
                    new_cost = cost + g.get_edge_data(node, n)['weight']
                    if new_cost < costs.get(n, self.infinity):
                        costs[n] = new_cost
                        hop[n] = hop[node] + 1

                        if hop[n] != 0 and (hop[n] - 1) % k == 0:
                            v_to_add.add(n)
                        processed.append(node)
                r_tmp.append(costs[node])
                node = find_lowest_cost_node(costs, processed)

            for v in v_to_add:
                g.add_edge(s, v, weight=costs[v])
            r[s] = max(r_tmp)


        pool = Pool(processes=4)
        for i in g.nodes:
            pool.apply_async(greedy, (i, 10, 20, self.r))  # params: s, k, rho, r
            pool.close()
        pool.join()


        pass
        
    def radius_stepping(self, g):
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


        while len(self.Q) > 0:
            
            di = min(self.R.values())
            a = []
            for key in self.Q.keys():
                if self.Q[key] <= di:
                    a.append(key)
            
            for key in a:
                del self.Q[key]
                del self.R[key]

            while len(a) > 0:#

                node = a.pop() 
                for v in g.neighbors(node):
                    temp = self.property_map[node] + g.get_edge_data(node, v)['weight']
                    if self.property_map[v] > temp:
                        if self.property_map[v] > di and temp <= di:
                            if v in self.R:
                                del self.R[v]
                            if v in self.Q:
                                del self.Q[v]
                            a.append(v)
                        self.property_map[v] = temp
                        #print(self.property_map)
                        if self.property_map[v] > di:
                            self.Q[v] = self.property_map[v]
                            self.R[v] = self.property_map[v] + self.r[v]
        return

def main():

    g = nx.read_edgelist('New York_after.txt', nodetype=int, data=(('weight', int),))



    print(nx.info(g))

    a = RStepping()
    a.preprocess(g)
    start = time.time()
    a = RStepping()
    a.preprocess(g)
    end = time.time()
    print(end - start)
    start = time.time()


    a.radius_stepping(g)
    end = time.time()
    print(end - start)

    # a.radius_stepping(g)
    # print(a.property_map)

    #visualize the graph


if __name__ == '__main__':


    main()