import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import os

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        
    def predict(self, data):
        # Convert input to DataFrame with proper column names
        columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                  'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                  'MonthlyCharges', 'TotalCharges']
        
        df = pd.DataFrame(data, columns=columns)
        
        # Handle TotalCharges - ensure it's numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        
        # Separate categorical and numeric features
        cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                   'PaperlessBilling', 'PaymentMethod']
        
        num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        # One-hot encode categorical variables
        if cat_cols:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
            final_df = pd.concat([df[num_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        else:
            final_df = df[num_cols].copy()
        
        # Scale numeric features
        scaler = StandardScaler()
        final_df[num_cols] = scaler.fit_transform(final_df[num_cols])
        
        # Ensure all training columns exist (pad with zeros if missing)
        try:
            model_features = self.model.feature_names_in_
            for col in model_features:
                if col not in final_df.columns:
                    final_df[col] = 0
            
            # Select only model features in correct order
            final_df = final_df.reindex(columns=model_features, fill_value=0)
        except:
            # If feature_names_in_ is not available, use as is
            pass
        
        prediction = self.model.predict(final_df)
        
        return "Will Churn" if prediction[0] == 1 else "Will Not Churn"