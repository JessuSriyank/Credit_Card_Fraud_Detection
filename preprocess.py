import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(dataset_path):
    # Load the dataset
    dataset = pd.read_csv(dataset_path)
    dataset = dataset.drop(['Time'], axis=1)  # Remove unnecessary column
    dataset['Amount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1, 1))
    
    # Split the data
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train.values, X_test.values, y_train.values, y_test.values
