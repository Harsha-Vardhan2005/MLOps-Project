import pandas as pd
import os
from mlProject import logger
from sklearn.ensemble import RandomForestClassifier
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load train/test
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]

        # Train RandomForest
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        # Save model
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        logger.info(f"Model saved at {self.config.root_dir}/{self.config.model_name}")
