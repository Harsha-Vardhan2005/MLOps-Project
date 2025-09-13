import os
import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            # Load dataset
            data = pd.read_csv(self.config.unzip_data_dir)
            dataset_columns = list(data.columns)

            # Expected schema
            expected_columns = list(self.config.all_schema.keys())

            # Compare
            missing_cols = [col for col in expected_columns if col not in dataset_columns]
            extra_cols = [col for col in dataset_columns if col not in expected_columns]

            validation_status = True

            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                validation_status = False

            if extra_cols:
                logger.warning(f"Extra columns found: {extra_cols} (will be ignored if not needed)")
                # You can decide to set validation_status = False here if strict check is required

            # Write result
            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data validation completed with status: {validation_status}")
            return validation_status

        except Exception as e:
            raise e
