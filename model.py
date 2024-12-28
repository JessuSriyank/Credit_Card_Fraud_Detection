import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_model(X_train):
    input_layer = Input(shape=(X_train.shape[1],))
    encoder = Dense(14, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(7, activation="relu")(encoder)
    decoder = Dense(7, activation="tanh")(encoder)
    decoder = Dense(X_train.shape[1], activation="relu")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.2, shuffle=True)

    return autoencoder

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = np.mean(np.power(X_test - predictions, 2), axis=1)
    threshold = 2.9
    y_pred = [1 if e > threshold else 0 for e in mae]
    return accuracy_score(y_test, y_pred)

def plot_results():
    # Example placeholder for visualizations
    print("Plotting results...")
