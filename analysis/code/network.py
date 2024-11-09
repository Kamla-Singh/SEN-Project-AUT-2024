import networkx as nx
from node import Node


class Network:
    def __init__(self, num_nodes, avg_degree, data):
        self.graph = nx.watts_strogatz_graph(
            num_nodes, avg_degree, (avg_degree+1)/num_nodes)
        self.nodes = [Node(node_id, data, num_nodes) for node_id in range(num_nodes)]

        # Create edges between nodes based on network structure
        for u, v in self.graph.edges():
            self.nodes[u].add_neighbor(self.nodes[v])
            self.nodes[v].add_neighbor(self.nodes[u])

    def get_nodes(self):
        return self.nodes
