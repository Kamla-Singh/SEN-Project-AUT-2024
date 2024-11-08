from data_reader import load_data
from network import Network
from simulator import Simulator


def main():
    datasets = {
        "Red Wine Quality": "data/datasets/WineQuality/winequality-red.csv",
        "White Wine Quality": "data/datasets/WineQuality/winequality-white.csv",
        "Online News Popularity": "data/datasets/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    }
    for data_name in datasets:
        print(f"\n{'=' * 100}\nAnalysis for {data_name}\n{'=' * 100}")
        X_train, X_test, y_train, y_test = load_data(
            file_path=datasets[data_name])
        data = (X_train, y_train)  # Assign train data to nodes

        # Set up the network with varying average degrees
        # Change this for different network structures
        num_nodes = 50
        avg_degree = 45
        network = Network(num_nodes=num_nodes,
                          avg_degree=avg_degree, data=data)

        # Run the simulation
        rounds = 1
        simulator = Simulator(network=network, rounds=rounds)
        simulator.run()


if __name__ == "__main__":
    main()
