from data_reader import load_data
from network import Network
from simulator import Simulator


def main():
    datasets = {
        "Red Wine Quality": "data/datasets/WineQuality/winequality-red.csv",
        "White Wine Quality": "data/datasets/WineQuality/winequality-white.csv",
        "Online News Popularity": "data/datasets/OnlineNewsPopularity/OnlineNewsPopularity.csv",
        "Daily Demand Forecasting Orders": "data/datasets/DailyDemandForecastingOrders/Daily_Demand_Forecasting_Orders.csv",
    }
    for data_name in datasets:
        print(f"\n{'=' * 100}\nAnalysis for {data_name}\n{'=' * 100}")
        X_train, X_test, y_train, y_test = load_data(
            file_path=datasets[data_name])
        data = (X_train, y_train)  # Assign train data to nodes

        # Set up the network with varying average degrees
        # Change this for different network structures
        num_nodes = 20
        efficient_network = Network(num_nodes=num_nodes,
                                    avg_degree=19, data=data)
        inefficient_network = Network(num_nodes=num_nodes,
                                      avg_degree=4, data=data)
        print("Efficient network", "\n\tNumber of Nodes:",
              num_nodes, "\n\tAverage Degree:", 19)
        print("Inefficient network", "\n\tNumber of Nodes:",
              num_nodes, "\n\tAverage Degree:", 4)

        # Run the simulation
        rounds = 16
        simulator = Simulator(
            efficient_network=efficient_network, inefficient_network=inefficient_network, data_name=data_name, rounds=rounds)
        simulator.run()


if __name__ == "__main__":
    main()
