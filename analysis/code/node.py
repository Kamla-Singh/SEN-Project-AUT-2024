# node.py
import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_bic(solution, residual):
    n = len(residual)
    rss = np.sum(residual ** 2)
    return n * np.log(rss / n) + np.count_nonzero(solution) * np.log(n)


class Node:
    def __init__(self, node_id, data, num_nodes):
        self.node_id = node_id
        self.num_nodes = num_nodes
        # List of neighboring nodes
        self.neighbors = []
        # Local data for regression (X, y)
        self.X, self.y = data

        # Initial random solution
        self.solution = np.random.randn(self.X.shape[1])
        # Initialize residual
        self.residual = self.y - np.dot(self.X, self.solution)

    def add_neighbor(self, neighbor_node):
        self.neighbors.append(neighbor_node)

    def explore(self):
        best_variable = None
        best_improvement = float('inf')

        # Iterate over each variable to find the best partial regression
        for i in range(self.X.shape[1]):
            # Create a temporary model updating only i-th variable
            X_partial = self.X[:, i].reshape(-1, 1)
            partial_model = LinearRegression().fit(X_partial, self.residual +
                                                   self.X[:, i] * self.solution[i])
            new_coef = partial_model.coef_[0]
            # Calculate improvement in residual if variable `i` is updated
            new_solution = np.copy(self.solution)
            new_solution[i] = new_coef
            new_residual = self.y - np.dot(self.X, new_solution)
            improvement = calculate_bic(
                solution=new_solution, residual=new_residual)

            # Track the best improvement
            if improvement < best_improvement:
                best_improvement = improvement
                best_variable = i
                best_new_coef = new_coef

        # Update the solution and residual with the best variable's new coefficient
        if best_variable is not None and best_improvement < self.calculate_error():
            self.solution[best_variable] = best_new_coef
            self.residual = self.y - np.dot(self.X, self.solution)

    def exploit(self):
        # Track the best neighbor solution and its error
        best_neighbor_solution = None
        best_neighbor_error = float('inf')

        # Iterate over neighbors' solutions to find the best one
        for neighbor in self.neighbors:
            neighbor_error = calculate_bic(
                solution=neighbor.solution, residual=neighbor.residual)
            if neighbor_error < best_neighbor_error:
                best_neighbor_error = neighbor_error
                best_neighbor_solution = neighbor.solution

        # Only copy the best neighbor's solution if it improves upon the current solution
        if best_neighbor_solution is not None and best_neighbor_error < self.calculate_error():
            self.solution = best_neighbor_solution
            self.residual = self.y - np.dot(self.X, self.solution)

    def calculate_error(self):
        return calculate_bic(solution=self.solution, residual=self.residual)

    def choose_action(self, roundq_num):
        # Decide whether to explore or exploit with some probability say 50%
        if np.random.rand() < len(self.neighbors) / self.num_nodes :
            self.explore()
        else:
            self.exploit()
