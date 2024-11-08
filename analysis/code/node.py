# node.py
import numpy as np


class Node:
    def __init__(self, node_id, data):
        self.node_id = node_id
        self.data = data  # Local data for regression (X, y)
        self.weights = np.random.randn(
            data[0].shape[1])  # Initial random weights
        self.neighbors = []  # List of neighboring nodes

    def add_neighbor(self, neighbor_node):
        self.neighbors.append(neighbor_node)

    def explore(self):
        # Update weights by randomly adjusting them (explore step)
        adjustment = np.random.normal(0, 0.1, size=self.weights.shape)
        self.weights += adjustment

    def exploit(self, neighbor_weights):
        # Copy (exploit) the neighbor's solution
        self.weights = neighbor_weights

    def calculate_error(self):
        X, y = self.data
        predictions = X @ self.weights
        return ((predictions - y) ** 2).mean()  # Mean squared error

    def choose_action(self, round_num):
        # Decide whether to explore or exploit with some probability say 50%
        if np.random.rand() < 0.5:
            self.explore()
        else:
            if self.neighbors:
                # Exploit by copying weights from a neighbor with minimum error
                neighbor = min(
                    self.neighbors, key=lambda n: n.calculate_error())
                self.exploit(neighbor.weights)
