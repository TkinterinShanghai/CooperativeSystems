import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import OrderedDict, deque

class Primary:
    def __init__(self, size):
        self.columns = size
        self.rows = size

        self.network = np.full((size, size), 0)  # Square Matrix
        self.fill_network(size)

        self.G = nx.from_numpy_matrix(self.network)
        self.primary = dict(nx.all_pairs_dijkstra(self.G))
        self.alternate = {}

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
        # nx.draw_networkx_edges(self.G, edgelist=([color_changes]),pos=layout, edge_color='r')
        nx.draw_networkx_edge_labels(self.G, pos=layout, edge_labels=labels)
        plt.show()

    def alternate_next(self):
        P_i = {"alt_next_hops": {-1}, "alt_type": None, "alt_link_protect": False, "alt_node_protect": False} #Primary next hops, default
        H_i = {"cand_type": "Loop-free", "cand_link_protect": False, "cand_node_protect": False} #Alternate next hops, default

        for start in self.rows:
            self.alternate[start] = []
            for destination in self.columns:
                if start == destination:
                    continue
                next_hop = P_i
                D_opt_S_D = self.primary[start][0][destination]    # Ideal distance from start to dest according to Dijkstra
                for alt_next_hop in self.G.neighbors(start): # Looping through each neighbor
                    primary_next_hop = self.primary[start][1][destination][1]
                    if alt_next_hop != primary_next_hop:  # Step 2
                        D_opt_H_D = self.primary[alt_next_hop][0][destination]  # Ideal distance from neighbor to destination
                        D_opt_H_S = self.primary[alt_next_hop][0][start]  # Distance from neighbor to start
                        if D_opt_H_D < D_opt_H_S + D_opt_S_D:  # Step 4, Step 3 is assumed
                            candidate = H_i  # Step 5, H_i is loop free by default,
                            if D_opt_H_S + D_opt_H_D == D_opt_S_D:
                                candidate["cand_type"] = "Primary"  # Step 6
                            # Step 7 Omitted because no shared link is implemented
                            D_opt_H_P = self.primary[alt_next_hop][0][primary_next_hop]
                            D_opt_P_D = self.primary[primary_next_hop][0][destination]
                            if D_opt_H_D < D_opt_H_P + D_opt_P_D:
                                candidate["cand_node_protect"] = True  # Step 8
                            # Step 9 Omitted because SRLG not considered
                            if candidate["cand_type"] == "Primary" and next_hop["alt_type"] != "Primary": # Step 10
                                next_hop["alt_next_hops"] = {alt_next_hop}          # Step 20
                                next_hop["alt_type"] = candidate["cand_type"]
                                next_hop["alt_node_protect"] = candidate["cand_node_protect"]
                                continue
                            elif candidate["cand_type"] != "Primary" and next_hop["alt_type"] == "Primary":  # Step 11
                                continue
                            if candidate["cand_node_protect"] == True and next_hop["alt_node_protect"] == False: # Step 12
                                next_hop["alt_next_hops"] = {alt_next_hop}          # Step 20
                                next_hop["alt_type"] = candidate["cand_type"]
                                next_hop["alt_node_protect"] = candidate["cand_node_protect"]
                            # Step 13 Omitted because no shared link is implemented
                            # Step 14 Omitted because SRLG not considered
                            # Step 15 Omitted because SRLG not considered
                            D_opt_ALT_D = self.primary[next_hop["alt_next_hop"]][0][destination]
                            if D_opt_H_D < D_opt_P_D and D_opt_ALT_D >= D_opt_P_D:  # Step 16
                                next_hop["alt_next_hops"] = {alt_next_hop}          # Step 20
                                next_hop["alt_type"] = candidate["cand_type"]
                                next_hop["alt_node_protect"] = candidate["cand_node_protect"]
                            if D_opt_H_D <= D_opt_ALT_D:  # Step 17, if the distance from the candidate to destination is shorter than the alternate next hop, it is being preferred
                                next_hop["alt_next_hops"] = {alt_next_hop}          # Step 20
                                next_hop["alt_type"] = candidate["cand_type"]
                                next_hop["alt_node_protect"] = candidate["cand_node_protect"] 
                            # Step 18 Continue to next alt_next_hop
                            # Step 19 

myNetwork = Primary(size=9)
myNetwork.print_network()
myNetwork.plot_network()

myNetwork.primary[2]