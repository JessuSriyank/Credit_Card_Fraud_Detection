from preprocess import preprocess_data
from model import train_model, evaluate_model, plot_results

if __name__ == "__main__":
    # Step 1: Load and preprocess the dataset
    dataset_path = "data/dataset.csv"
    X_train, X_test, y_train, y_test = preprocess_data(dataset_path)

    # Step 2: Train the model
    autoencoder = train_model(X_train)

    # Step 3: Evaluate the model
    accuracy = evaluate_model(autoencoder, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Step 4: Generate visualizations
    plot_results()
