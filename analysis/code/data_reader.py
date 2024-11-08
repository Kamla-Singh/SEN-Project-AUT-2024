import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path).select_dtypes(include=['number'])

    # Assuming the target column is the last one
    X = data.iloc[:, :-1].values  # All columns except the last as features
    y = data.iloc[:, -1].values   # Last column as the target

    # Preprocess data (e.g., scaling)
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
