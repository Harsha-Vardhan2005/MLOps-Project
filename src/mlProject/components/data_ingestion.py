import os
import shutil
from mlProject import logger
from pathlib import Path
from mlProject.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        """
        Instead of downloading/unzipping, 
        just copy the local dataset into the artifacts directory.
        """
        # Ensure target directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Copy dataset to artifacts folder
        if os.path.exists(self.config.local_data_file):
            target_path = os.path.join(self.config.root_dir, "Churn.csv")
            shutil.copy(self.config.local_data_file, target_path)
            logger.info(f"Dataset copied to {target_path}")
        else:
            raise FileNotFoundError(f"Dataset not found at {self.config.local_data_file}")
