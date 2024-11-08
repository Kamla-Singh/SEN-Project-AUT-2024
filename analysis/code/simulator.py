import numpy as np


class Simulator:
    def __init__(self, network, rounds=10):
        self.network = network
        self.rounds = rounds

    def run(self):
        print(f"{'Round':<10} {'Avg Error':<15} {
              'Min Error':<15} {'Max Error':<15}")
        print("-" * 100)
        for round_num in range(self.rounds):
            # permute the order of nodes in each round
            for node in np.random.permutation(self.network.get_nodes()):
                node.choose_action(round_num)
            self._log_round_results(round_num)

    def _log_round_results(self, round_num):
        # Logging results for each round
        # Calculate average error across all nodes
        avg_error = np.mean([node.calculate_error()
                             for node in self.network.get_nodes()])
        # Calculate least error across all nodes
        min_error = min([node.calculate_error()
                         for node in self.network.get_nodes()])
        # Calculate maximum error across all nodes
        max_error = max([node.calculate_error()
                         for node in self.network.get_nodes()])
        print(f"{round_num:<10} {avg_error:<15.3f} {
              min_error:<15.3f} {max_error:<15.3f}")
