import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import collections


class Primary:
    def __init__(self, size, seed):
        self.columns = size
        self.rows = size

        self.seed = seed

        self.network = np.full((size, size), 0)  # Square Matrix
        self.fill_network(size)

        self.G = nx.from_numpy_matrix(self.network)
        self.primary = dict(nx.all_pairs_dijkstra(self.G))

        self.lfa_frr_paths = {}
        self.ti_lfa_paths = {}

    def fill_network(self, size):
        columns = size
        rows = size
        mincolumn = 1
        np.random.seed(self.seed)
        for row in range(rows):
            for column in range(mincolumn, columns):
                link_weight = np.random.choice([0, np.random.randint(1, 10)], p=[0.5, 0.5])
                self.network[row, column] = link_weight
                self.network[column, row] = link_weight
            mincolumn += 1
        # self.network[0] = [0, 1, 0, 0, 0, 1]
        # self.network[1] = [1, 0, 1, 0, 0, 0]
        # self.network[2] = [0, 1, 0, 1, 0, 0]
        # self.network[3] = [0, 0, 1, 0, 2, 0]
        # self.network[4] = [0, 0, 0, 2, 0, 1]
        # self.network[5] = [1, 0, 0, 0, 1, 0]

    def print_network(self):
        print("  ", end='')
        for i in range(self.columns):
            print(f" {i}", end=' ')
        print('')
        for i in range(self.rows):
            print(i, end=' ')
            print(self.network[i])

    def print_results(self, alternate):
        for start, alt_next_hops in alternate.items():
            print(f"########{start}##########")
            for destination, alt_next_hop in alt_next_hops.items():
                print(f"{destination}: {alt_next_hop}")

    def plot_network(self):
        layout = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, layout, width=0.3)
        labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos=layout, edge_labels=labels)
        plt.show()

    def plot_results(self, start, destination):
        layout = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, layout, width=0.3)
        labels = nx.get_edge_attributes(self.G, "weight")
        primary_route = self.primary[start][1][destination]

        primary_colors = [(start, destination) for start, destination in zip(primary_route, primary_route[1:])]
        nx.draw_networkx_edges(self.G, edgelist=primary_colors, pos=layout, edge_color='blue',
                               label='Primary Route', width=3)

        alternate_next_hop = self.lfa_frr_paths[start][destination]['alt_next_hops']
        if  alternate_next_hop != -1 and self.lfa_frr_paths[start]:
            alternate_route = [start] + self.primary[alternate_next_hop][1][destination]
            lfa_frr_colors = [(start, destination) for start, destination in zip(alternate_route, alternate_route[1:])]
            nx.draw_networkx_edges(self.G, edgelist=lfa_frr_colors, pos=layout, edge_color='r',
                                   label='LFA FRR', width=3)

        segments = self.ti_lfa_paths[start][destination]
        for segment in segments:
            if segment['along-convergence']:
                alternate_route = self.primary[start][1][segment['node'][0]]
                for node in segment['node'][1:]:
                    alternate_route.append(node)
                if len(segment['node']) != 1:
                    alternate_route += self.primary[segment['node'][-1]][1][destination]
                ti_lfa_colors = [(start, dest) for start, dest in zip(alternate_route, alternate_route[1:])]
                print(ti_lfa_colors)
                nx.draw_networkx_edges(self.G, edgelist=ti_lfa_colors, pos=layout, edge_color='g',
                                       label='TI LFA', width=3, style='dotted')
                break

        plt.legend()
        nx.draw_networkx_edge_labels(self.G, pos=layout, edge_labels=labels)
        plt.show()


    def frr_lfa(self):
        D_opt_ALT_D = math.inf
        P_i = {"alt_next_hops": -1, "alt_type": None, "alt_link_protect": False,
               "alt_node_protect": False}  # Primary next hops, default
        H_i = {"cand_value": -1, "cand_type": "Loop-free", "cand_link_protect": False,
               "cand_node_protect": False}  # Alternate next hops, default

        for start in range(self.rows):
            self.lfa_frr_paths[start] = {}
            for destination in range(self.columns):
                if start == destination:
                    continue
                next_hop = P_i.copy()
                try:
                    D_opt_S_D = self.primary[start][0][
                        destination]  # Ideal distance from start to dest according to Dijkstra
                except KeyError:  # Node is not connected to any other node in the network
                    continue
                primary_next_hop = self.primary[start][1][destination][1]
                for alt_next_hop in self.G.neighbors(start):  # Looping through each neighbor
                    if alt_next_hop != primary_next_hop:  # Step 2
                        D_opt_H_D = self.primary[alt_next_hop][0][
                            destination]  # Ideal distance from neighbor to destination
                        D_opt_H_S = self.primary[alt_next_hop][0][start]  # Distance from neighbor to start
                        if D_opt_H_D < D_opt_H_S + D_opt_S_D:  # Step 4, Step 3 is assumed
                            candidate = H_i.copy()  # Step 5, H_i is loop free by default,
                            candidate["cand_value"] = alt_next_hop
                            if self.network[start][alt_next_hop] + D_opt_H_D == D_opt_S_D:
                                candidate["cand_type"] = "Primary"  # Step 6
                            candidate[
                                "cand_link_protect"] = True  # Step 7, no shared links, so if-statement always satisfied
                            D_opt_H_P = self.primary[alt_next_hop][0][primary_next_hop]
                            D_opt_P_D = self.primary[primary_next_hop][0][destination]
                            if D_opt_H_D < D_opt_H_P + D_opt_P_D:
                                candidate["cand_node_protect"] = True  # Step 8
                            # Step 9 Omitted because SRLG not considered
                            if candidate["cand_type"] == "Primary" and next_hop["alt_type"] != "Primary":  # Step 10
                                next_hop = dict(zip(next_hop.keys(), candidate.values()))  # Step 20
                                D_opt_ALT_D = D_opt_H_D
                                continue
                            elif candidate["cand_type"] != "Primary" and next_hop["alt_type"] == "Primary":  # Step 11
                                continue
                            if candidate["cand_node_protect"] == True and next_hop[
                                "alt_node_protect"] == False:  # Step 12
                                next_hop = dict(zip(next_hop.keys(), candidate.values()))  # Step 20
                                D_opt_ALT_D = D_opt_H_D
                                continue
                            if candidate["cand_link_protect"] == True and next_hop[
                                "alt_link_protect"] == False:  # Step 13
                                next_hop = dict(zip(next_hop.keys(), candidate.values()))  # Step 20
                                D_opt_ALT_D = D_opt_H_D
                                continue
                            # Step 14 Omitted because SRLG not considered
                            # Step 15 Omitted because SRLG not considered
                            if D_opt_H_D < D_opt_P_D and D_opt_ALT_D >= D_opt_P_D:  # Step 16
                                next_hop = dict(zip(next_hop.keys(), candidate.values()))  # Step 20
                                D_opt_ALT_D = D_opt_H_D
                                continue
                            if D_opt_H_D < D_opt_ALT_D:  # Step 17, if the distance from the candidate to destination is shorter than the alternate next hop, it is being preferred
                                next_hop = dict(zip(next_hop.keys(), candidate.values()))  # Step 20
                                D_opt_ALT_D = D_opt_H_D
                                continue
                            # Step 18 Continue to next alt_next_hop
                            # Step 19 
                self.lfa_frr_paths[start][destination] = next_hop

    def ti_lfa(self):
        for start in range(self.rows):
            self.ti_lfa_paths[start] = {}
            for destination in range(self.columns):
                if start == destination:
                    continue

                primary_next_hop = self.primary[start][1][destination][1]

                # calculating the post-convergence path, www.segment-routing.net page 33
                weight_to_primary = self.G[start][primary_next_hop]['weight']
                self.G.remove_edge(start, primary_next_hop)
                if self.G[start] == {}:  # no alternate next hops available for selected start
                    self.G.add_edge(start, primary_next_hop, weight=weight_to_primary)  # origina graph is restored
                    break
                try:
                    post_convergence_path = nx.dijkstra_path(self.G, start, destination)
                except nx.exception.NetworkXNoPath:
                    self.G.add_edge(start, primary_next_hop, weight=weight_to_primary)
                    continue
                self.G.add_edge(start, primary_next_hop, weight=weight_to_primary)

                self.ti_lfa_paths[start][destination] = []
                p_space = collections.defaultdict(dict)
                q_space = collections.defaultdict(dict)

                for node in range(self.columns):
                    path_to_node = self.primary[start][1][node][:2]
                    if path_to_node != [start, primary_next_hop]:
                        p_space[node]['link-protect'] = True
                    else:
                        continue
                    if primary_next_hop not in self.primary[start][1][node]:
                        p_space[node]['node-protect'] = True

                for node in range(self.columns):
                    path_to_destination = self.primary[node][1][destination]
                    if ', '.join(map(str, [start, primary_next_hop])) not in ', '.join(
                            map(str, path_to_destination)):
                        q_space[node]['link-protect'] = True
                    else:
                        continue
                    if primary_next_hop not in path_to_destination:
                        q_space[node]['node-protect'] = True

                for node in p_space.keys():
                    if node in q_space.keys():
                        segment = {'node': [node], 'along-convergence': False, 'link-protect': False,
                                   'node-protect': False}
                        if node in post_convergence_path:
                            segment['along-convergence'] = True
                        if 'link-protect' in p_space[node].keys() and 'link-protect' in q_space[node].keys():
                            segment['link-protect'] = True
                        if 'node-protect' in p_space[node].keys() and 'node-protect' in q_space[node].keys():
                            segment['node-protect'] = True
                        if segment['node-protect'] or segment['link-protect']:  # path has to be link or node protected
                            self.ti_lfa_paths[start][destination].append(segment)

                    # if there is no overlap in the P and Q space, check whether
                    # P and Q space are adjacent to each other or can be connected via pushing segments:
                    if not self.ti_lfa_paths[start][destination]:
                        segment = {'node': None, 'along-convergence': True, 'link-protect': False,
                                   'node-protect': False}
                        last_in_p_space = {}
                        between_q_and_p = []
                        for node in post_convergence_path:
                            if node in p_space:
                                last_in_p_space = {'node': node, 'along-convergence': True, 'link-protect': False,
                                                   'node-protect': False}
                                last_in_p_space.update(p_space[node])
                                continue
                            elif node in q_space:  # once the first node in the q space is reached, the loop is over
                                first_in_q_space = {'node': node, 'along-convergence': True, 'link-protect': False,
                                                    'node-protect': False}
                                first_in_q_space.update(q_space[node])
                                segment['node'] = [last_in_p_space['node'], first_in_q_space['node']]
                                if between_q_and_p:
                                    segment['node'].insert(1, between_q_and_p)
                                if last_in_p_space['link-protect'] and first_in_q_space['link-protect']:
                                    segment['link-protect'] = True
                                if last_in_p_space['node-protect'] and first_in_q_space['node-protect']:
                                    segment['node-protect'] = True
                                if segment['node-protect'] or segment['link-protect']:  # path has to be protected
                                    self.ti_lfa_paths[start][destination].append(segment)
                                break
                            else:  # q space is not reached yet. the segments inbetween are added to the list
                                between_q_and_p.append(node)
                                continue


myNetwork = Primary(size=6, seed=15)

myNetwork.print_network()
myNetwork.plot_network()

myNetwork.frr_lfa()
myNetwork.print_results(myNetwork.lfa_frr_paths)

myNetwork.ti_lfa()
myNetwork.print_results(myNetwork.ti_lfa_paths)

myNetwork.plot_results(1,0)


