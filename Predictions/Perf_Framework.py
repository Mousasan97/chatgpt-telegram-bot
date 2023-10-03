from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from flask import Flask, request, jsonify
import json
import joblib
from tensorflow.keras.models import load_model

# Function Definitions
class ModelInference:
    def __init__(self, model_path, scaler_path, weather_data_path, actual_generation_path):
        self.load_model_and_scaler(model_path, scaler_path)
        self.weather_df = self.load_weather_data(weather_data_path)
        self.forecast_steps = 1
        self.y_test = self.load_actual_gen(actual_generation_path)

    def load_actual_gen(self, parquet_path):
        """Load actual generation data from a .parquet file."""
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        return df
           
    def load_weather_data(self, parquet_path):
        """Load weather data from a .parquet file."""
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        return df

    def r_squared(self, y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res/(SS_tot + K.epsilon()))
        
    
    def load_model_and_scaler(self, model_path, scaler_path):
        self.model = load_model(model_path, custom_objects={'r_squared': self.r_squared})
        self.scaler = joblib.load(scaler_path)


    def create_multistep_dataset(self, X, input_steps=1):
        Xs = []
        for i in range(input_steps, len(X) + 1):
            Xs.append(X.iloc[i-input_steps:i].values)
        return np.array(Xs)

    def prepare_data(self, input_steps=10):
        # Here, we now directly use the internally loaded weather_df 
        X_multistep = self.create_multistep_dataset(self.weather_df, input_steps=input_steps)
        self.X_new_data = X_multistep
        self.X_new_data = self.X_new_data.squeeze(axis=2)

    def predict(self):
        # Scale the prepared data using the loaded scaler
        X_scaled = self.scaler.transform(self.X_new_data)
        self.predictions = self.model.predict(X_scaled, verbose=0)
        return self.predictions
    
    def receive_actual_gen(self):
        self.generation = self.y_test
        return self.generation
    

    def detect_outliers(self):
        # Assuming y_test and y_pred are instance variables
        self.y_pred = self.predictions.copy()
        self.y_test = self.y_test.values
        self.residuals = np.abs(self.y_test - self.y_pred)
        # Convert the NumPy array to a DataFrame
        residual_df = pd.DataFrame(self.residuals, columns=['res'])
        # Calculate the threshold
        threshold = np.percentile(self.residuals, 90)  # Example: 99.5th percentile
        residual_df['is_outlier'] = residual_df['res'] > threshold
        residual_df.set_index(self.weather_df.index[-len(residual_df):], inplace=True)
        self.outliers = residual_df[residual_df['is_outlier'] == True]
        return self.residuals, self.outliers
                
           

        