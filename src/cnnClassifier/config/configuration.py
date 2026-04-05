import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_num_classes=self.params.NUM_CLASSES,
            params_dense_1=self.params.DENSE_1,
            params_dense_2=self.params.DENSE_2,
            params_dropout_1=self.params.DROPOUT_1,
            params_dropout_2=self.params.DROPOUT_2
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        prepare_base_model = self.config.prepare_base_model
        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )
        create_directories([config.root_dir])
        return TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs_phase1=self.params.PHASE1_EPOCHS,
            params_epochs_phase2=self.params.PHASE2_EPOCHS,
            params_lr_phase1=self.params.PHASE1_LR,
            params_lr_phase2=self.params.PHASE2_LR,
            params_batch_size=self.params.BATCH_SIZE,
            params_image_size=self.params.IMAGE_SIZE,
            params_classes=self.params.CLASSES,
            params_seed=self.params.SEED,
            params_fine_tune_layers=self.params.FINE_TUNE_LAYERS,
            params_class_weights={int(k): v for k, v in self.params.CLASS_WEIGHTS.items()}
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir),
            mlflow_uri=self.config.evaluation.mlflow_uri,
            all_params=dict(self.params),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_classes=self.params.CLASSES
        )