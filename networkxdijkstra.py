import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

    # https://gist.github.com/krisprice/31eda711ebd5015396c72e9d29ab0c60
    def apply_ecmp_flow(self, source, destination):
        try:
            paths = list(nx.all_shortest_paths(self.G, source, destination, weight='weight'))
            num_ecmp_paths = len(paths)

            for p in paths:
                u = p[0]
                for v in p[1:]:
                    min_weight = min(d['weight'] for d in self.G[u][v].values())
                    keys = [k for k, d in self.G[u][v].items() if d['weight'] == min_weight]
                    num_ecmp_links = len(keys)

                    # for k in keys:
                    #     self.G[u][v][k]['total_utilization'] += load / num_ecmp_paths / num_ecmp_links
                    u = v
        except (KeyError, nx.NetworkXNoPath):
            print("Error, no path for %s to %s in apply_ecmp_flow()" % (source, destination))
            raise

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
        P_i = {"alt_next_hops": {}, "alt_link": False, "alt_node": False} #Primary next hops, default
        H_i = {"cand_type": "Loop-free", "cand_link_protect": False, "cand_node_protect": False} #Alternate next hops, default

        for start in self.rows:
            self.alternate[start] = []
            for destination in self.columns:
                D_opt_S_D = self.primary[start][0][destination]    # Ideal distance from start to dest according to Dijkstra
                for alt_next_hop in self.G.neighbors(start): # Looping through each neighbor
                    primary_next_hop = self.primary[start][1][destination][1]
                    if alt_next_hop != primary_next_hop:  # Step 2
                        D_opt_H_D = self.primary[alt_next_hop][0][destination]  # Ideal distance from neighbor to destination
                        D_opt_H_S = self.primary[alt_next_hop][0][start]  # Distance from neighbor to start
                        if D_opt_H_D <= D_opt_H_S + D_opt_S_D:  # Step 4, Step 3 is assumed
                            candidate = H_i  # Step 5, H_i is loop free by default,
                            # Step 6 Omitted because ECMP not implemented yet
                            # Step 7 Omitted               "
                            # for primary in self.primary[start][1][destination]:
                            D_opt_H_P = self.primary[alt_next_hop][0][primary_next_hop]
                            D_opt_P_D = self.primary[primary_next_hop][0][destination]
                            if D_opt_H_D < D_opt_H_P + D_opt_P_D:
                                candidate["cand_node_protect"] = True  # Step 8
                                # Step 9 Omitted because SRLG not considered
                                # Step 10 Omitted
                                # Step 11 ???
                                # self.primary[]

                                # if D_opt_H_D

myNetwork = Primary(size=8)
myNetwork.print_network()
myNetwork.plot_network()

print(myNetwork.primary[2])

myNetwork.apply_ecmp_flow(1,2)
