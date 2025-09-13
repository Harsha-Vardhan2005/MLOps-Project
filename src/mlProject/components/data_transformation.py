import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        # Load dataset
        data = pd.read_csv(self.config.data_path)

        # Drop ID columns if present
        drop_cols = ["customerID"] if "customerID" in data.columns else []
        data.drop(columns=drop_cols, inplace=True, errors="ignore")

        # Handle TotalCharges column (common issue in telecom dataset)
        if 'TotalCharges' in data.columns:
            # Convert TotalCharges to numeric, replace empty strings with 0
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)

        # Separate features and target
        target_col = self.config.target_column
        X = data.drop(columns=[target_col])
        y = data[target_col].apply(lambda x: 1 if x == "Yes" else 0)  # Encode target as 0/1

        # Identify categorical and numeric features
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Handle missing values in numeric columns
        if num_cols:
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())

        # One-hot encode categorical
        if cat_cols:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(X[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
            # Reset index to ensure proper concatenation
            encoded_df.index = X.index
            X = pd.concat([X[num_cols], encoded_df], axis=1)
        
        # Scale numeric features
        if num_cols:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Save to CSVs
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        # Ensure proper index alignment before concatenation
        train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info(f"Data transformed and saved at {self.config.root_dir}")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        # Check for NaN values
        if train.isnull().sum().sum() > 0:
            logger.warning(f"Train data contains {train.isnull().sum().sum()} NaN values")
        if test.isnull().sum().sum() > 0:
            logger.warning(f"Test data contains {test.isnull().sum().sum()} NaN values")