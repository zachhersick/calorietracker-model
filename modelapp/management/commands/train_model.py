import os
from pathlib import Path
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Command(BaseCommand):
    help = 'Train the calorie prediction model using the dataset in src/main/resources'

    def handle(self, *args, **options):
        repo_root = Path(__file__).resolve().parents[3]
        csv_path = repo_root / 'src' / 'main' / 'resources' / 'diet_recommendations_dataset.csv'

        if not csv_path.exists():
            self.stderr.write(f"Dataset not found at {csv_path}. Please place the CSV there.")
            return

        df = pd.read_csv(csv_path)

        # Drop columns that are not used in the original script
        drop_cols = ['Patient_ID', 'BMI', 'Disease_Type', 'Severity', 'Cholesterol_mg/dl',
                     'Blood_Pressure_mmHg', 'Glucose_mg/dL', 'Dietary_Restrictions',
                     'Allergies,Preferred_Cuisine', 'Weekly_Exercise_Hours',
                     'Adherence_to_Diet_Plan', 'Dietary_Nutrient_Imbalance_Score',
                     'Diet_Recommendation']
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

        # One-hot encode categorical columns if present
        if 'Gender' in df.columns and 'Physical_Activity_Level' in df.columns:
            df = pd.get_dummies(df, columns=['Gender', 'Physical_Activity_Level'], drop_first=True)

        if 'Daily_Caloric_Intake' not in df.columns:
            self.stderr.write('CSV does not contain Daily_Caloric_Intake column')
            return

        X = df.drop('Daily_Caloric_Intake', axis=1)
        y = df['Daily_Caloric_Intake']

        if X.shape[1] == 0:
            self.stderr.write('No feature columns found after processing the CSV.')
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        output_dir = repo_root / 'modelapp' / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / 'model.h5'
        model.save(model_path)
        self.stdout.write(f'Model saved to {model_path}')

        # Evaluate
        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.stdout.write(f'MSE: {mse:.4f}, R2: {r2:.4f}')

        # Plot training history
        plt.figure()
        plt.plot(history.history.get('loss', []), label='Training Loss')
        plt.plot(history.history.get('val_loss', []), label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_plot = output_dir / 'loss.png'
        plt.savefig(loss_plot)
        plt.close()

        self.stdout.write(f'Plots saved to {output_dir}')
