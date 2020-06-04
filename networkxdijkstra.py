import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import OrderedDict, deque


class Network:
    def __init__(self, size):
        self.columns = size
        self.rows = size

        self.network = np.full((size, size), 0)  # Square Matrix
        self.fill_network(size)

        self.G = nx.from_numpy_matrix(self.network)

        self.nodes = np.empty(size, dtype=object)

    def fill_network(self, size):
        columns = size
        rows = size
        mincolumn = 1
        for row in range(rows):
            for column in range(mincolumn, columns):
                link_weight = np.random.choice([0, np.random.randint(1, 10)], p=[0.5, 0.5])
                self.network[row, column] = link_weight
                self.network[column, row] = link_weight
            mincolumn += 1

    def print_network(self):
        print("  ", end='')
        for i in range(self.columns):
            print(f" {i}", end=' ')
        print('')
        for i in range(self.rows):
            print(i, end=' ')
            print(self.network[i])

    def plot_network(self):
        layout = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, layout, width=0.3)
        labels = nx.get_edge_attributes(self.G, "weight")
        ### update this:
        color_changes = (self.network[0][1], self.network[0][2])
        # print("tuple:", color_changes)
        #nx.draw_networkx_edges(self.G, edgelist=([color_changes]),pos=layout, edge_color='r')
        nx.draw_networkx_edge_labels(self.G, pos=layout, edge_labels=labels)
        plt.show()
    def shortest_paths(self):
        len_path = dict(nx.all_pairs_dijkstra(self.G))
        print(len_path)
        # print(len_path[3][0][1])
        # for node in [0, 1, 2, 3, 4]:
        #     print('3 - {}: {}'.format(node, len_path[3][0][node]))
        # print(len_path[3][1][1])
        for n, (dist, path) in nx.all_pairs_dijkstra(self.G):
            print(path[1])

myNetwork = Network(size=5)
myNetwork.print_network()
myNetwork.plot_network()
myNetwork.shortest_paths()