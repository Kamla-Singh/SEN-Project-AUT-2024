import numpy as np
import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, efficient_network, inefficient_network, data_name, rounds=10):
        self.efficient_network = efficient_network
        self.inefficient_network = inefficient_network
        self.rounds = rounds
        self.data_name = data_name
        self.efficient_network_min_error = []
        self.efficient_network_avg_error = []
        self.inefficient_network_min_error = []
        self.inefficient_network_avg_error = []
        self.logger = open("analysis/results/" + data_name + ".txt", mode="w")

    def run(self):
        for round_num in range(self.rounds):
            self.logger.write(f"{'-' * 50} Round {round_num} {'-' * 50} \n")
            # permute the order of nodes in each round
            for node in np.random.permutation(self.efficient_network.get_nodes()):
                node.choose_action(round_num)
            for node in np.random.permutation(self.inefficient_network.get_nodes()):
                node.choose_action(round_num)
            self._log_round_results(round_num)
        self._plot_results()
        self.logger.close()

    def _log_round_results(self, round_num):
        # Logging results for each round
        efficient_network_error = {node.node_id: node.calculate_error()
                                   for node in self.efficient_network.get_nodes()}
        inefficient_network_error = {node.node_id: node.calculate_error()
                                     for node in self.inefficient_network.get_nodes()}
        self.logger.write("Efficient Network\n")
        for key, value in efficient_network_error.items():
            self.logger.write(f"Node {key}: {value}\n")
        self.logger.write("Inefficient Network\n")
        for key, value in inefficient_network_error.items():
            self.logger.write(f"Node {key}: {value}\n")
        self.efficient_network_min_error.append(
            np.min(list(efficient_network_error.values())))
        self.efficient_network_avg_error.append(
            np.mean(list(efficient_network_error.values())))
        self.inefficient_network_min_error.append(
            np.min(list(inefficient_network_error.values())))
        self.inefficient_network_avg_error.append(
            np.mean(list(inefficient_network_error.values())))

    def _plot_results(self):
        plt.figure(figsize=(10, 6))
        rounds = range(1, self.rounds+1)
        plt.plot(rounds, self.efficient_network_min_error, color='blue',
                 marker='o', label='Efficient Network')
        plt.plot(rounds, self.inefficient_network_min_error, color='red',
                 marker='o', label='Inefficient Network')
        plt.xlabel('Round Number')
        plt.ylabel('Minimum BIC')
        plt.title(f'Analysis of {self.data_name}')
        plt.legend()
        plt.savefig("report/figures/" + f"{self.data_name}" + " Min.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.efficient_network_avg_error, color='blue',
                 marker='o', label='Efficient Network')
        plt.plot(rounds, self.inefficient_network_avg_error, color='red',
                 marker='o', label='Inefficient Network')
        plt.xlabel('Round Number')
        plt.ylabel('Average BIC')
        plt.title(f'Analysis of {self.data_name}')
        plt.legend()
        plt.savefig("report/figures/" + f"{self.data_name}" + " Avg.png")
        plt.close()
